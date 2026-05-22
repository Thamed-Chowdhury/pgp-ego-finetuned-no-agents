"""
Full val-split zero-shot LLM trajectory selector.

For all 373 val (train_val) ego windows:
  1. Cache the model's K=10 trajectories + probs + GT (one inference pass on GPU).
  2. For each window with scene-level annotations, ask gemma-4-31b-it which of
     the 10 trajectory descriptions best matches each annotation.
  3. Per window, record:
       - top_conf_ADE      (current model)
       - LLM_picked_ADE    (LLM-selected, per annotation)
       - min_ADE           (oracle, best-of-K)
  4. Aggregate over windows: mean / per-bucket / improvement counts.

Resumable: inference cache + per-(window, annotation) JSONL append. Re-runs
skip already-classified pairs.

Usage:
  cd /teamspace/studios/this_studio
  python3 -u PGP_ego/llm_zero_shot_full.py
"""

import os, sys, json, pickle, time, re
import numpy as np
import pandas as pd
import torch
import yaml

os.environ.setdefault('RAY_memory_monitor_refresh_ms', '0')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_eval.initialization import initialize_prediction_model
import train_eval.utils as u
from nuscenes.eval.prediction.splits import create_splits_scenes, NUM_IN_TRAIN_VAL
import google.genai as genai
from google.genai import types as gtypes

from llm_zero_shot_selector import describe_trajectory, build_val_scene_pickle_map

# ── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT       = r'D:\DriveX_PGP\nuScenes_data'
PREPROC_DIR     = r'D:\DriveX_PGP\pgp-ego-prediction\pgp_ego_preprocessed'
ANNOT_CSV       = r'D:\DriveX_PGP\doScenes-VLM-Planning-main\data\doScenes\annotated_doscenes.csv'
KEYS_FILE       = r'D:\DriveX_PGP\Gemini_keys.txt'
CFG_FILE        = os.path.join(HERE, 'configs', 'pgp_ego_gatx2_lvm_traversal.yml')
CHECKPOINT      = r'D:\DriveX_PGP\pgp-ego-prediction\pgp_ego_output2\checkpoints\best.tar'
OUT_DIR         = r'D:\DriveX_PGP\pgp-ego-prediction\PGP_ego\documentation'
INFERENCE_CACHE = f'{OUT_DIR}/inference_cache.pkl'
RESULTS_JSONL   = f'{OUT_DIR}/full_val_llm_results.jsonl'

ANN_PRIORITY = {'d': 0, 'sd': 1, 'ds': 2, 's': 3, '?': 4}
MAX_ANN_PER_WINDOW = 3

CALL_GAP = 60.0 / 14   # ~4.3 s — under the 15 RPM per-key limit

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ── Inference cache ──────────────────────────────────────────────────────────
def build_or_load_inference_cache():
    if os.path.exists(INFERENCE_CACHE):
        print(f"Loading inference cache from {INFERENCE_CACHE} ...", flush=True)
        with open(INFERENCE_CACHE, 'rb') as f:
            return pickle.load(f)

    # Build cache from scratch
    with open(CFG_FILE) as f:
        cfg = yaml.safe_load(f)
    print("Loading ego model ...", flush=True)
    model = initialize_prediction_model(
        cfg['encoder_type'], cfg['aggregator_type'], cfg['decoder_type'],
        cfg['encoder_args'], cfg['aggregator_args'], cfg['decoder_args'],
    ).float().to(device)
    model.eval()
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.aggregator.num_samples = 100
    model.decoder.num_samples = 100
    print(f"  Loaded from {CHECKPOINT}", flush=True)

    print("Building val scene->pickle map ...", flush=True)
    scene_map = build_val_scene_pickle_map(PREPROC_DIR, DATA_ROOT)
    n_windows = sum(len(v) for v in scene_map.values())
    print(f"  {len(scene_map)} scenes, {n_windows} windows", flush=True)

    cache = {}   # (scene_num, pkl_path) → {'trajs', 'probs', 'gt', 'top_k', 'dists'}
    t0 = time.time()
    done = 0
    for sc, pkls in sorted(scene_map.items()):
        for pkl in pkls:
            with open(pkl, 'rb') as f:
                raw = pickle.load(f)
            data_t = u.send_to_device(u.convert_double_to_float(u.convert2tensors(raw)))
            with torch.no_grad():
                preds = model(data_t['inputs'])
            trajs = preds['traj'][0].detach().cpu().numpy()
            probs = preds['probs'][0].detach().cpu().numpy()
            gt    = data_t['ground_truth']['traj'][0].detach().cpu().numpy()
            dists = np.linalg.norm(trajs - gt[None], axis=-1).mean(axis=1)
            top_k = int(np.argmax(probs))
            cache[(sc, pkl)] = {
                'trajs': trajs, 'probs': probs, 'gt': gt,
                'dists': dists, 'top_k': top_k,
                'top_ade': float(dists[top_k]),
                'min_ade': float(dists.min()),
            }
            done += 1
            if done % 50 == 0:
                print(f"  Inference: {done}/{n_windows}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"  Inference done in {time.time()-t0:.0f}s", flush=True)

    with open(INFERENCE_CACHE, 'wb') as f:
        pickle.dump(cache, f)
    print(f"  Saved cache to {INFERENCE_CACHE}", flush=True)
    return cache


# ── Annotation handling ─────────────────────────────────────────────────────
def annotations_for_scene(df_val, scene_num, max_n):
    sub = df_val[df_val['scene_num'] == scene_num].copy()
    if len(sub) == 0:
        return []
    sub['priority'] = sub['ann_type'].map(lambda t: ANN_PRIORITY.get(t, 5))
    sub = sub.sort_values(['priority']).head(max_n)
    return sub.to_dict('records')


# ── Gemini selector ─────────────────────────────────────────────────────────
def make_prompt(annotation: str, descriptions: list) -> str:
    desc_block = "\n".join(f"  {i+1}. {d}" for i, d in enumerate(descriptions))
    return (
        "You are choosing the best predicted future trajectory for an autonomous "
        "vehicle. The trajectory must reflect the driver's stated intent.\n\n"
        f"Driver instruction:\n  \"{annotation}\"\n\n"
        f"The model produced 10 candidate 6-second trajectories:\n{desc_block}\n\n"
        "Which trajectory number (1-10) best matches the driver's intent? "
        "Consider direction (left/right/straight), maneuvers (turn, lane change, "
        "stop, slow down), and speed.\n"
        "Reply with ONLY the trajectory number — a single integer between 1 and 10."
    )


def find_working_keys(keys_file: str) -> list:
    with open(keys_file) as f:
        all_keys = [k.strip() for k in f if k.strip()]
    print(f"Pre-flight testing {len(all_keys)} keys ...", flush=True)
    working = []
    for i, k in enumerate(all_keys):
        c = genai.Client(api_key=k)
        for _ in range(2):
            try:
                c.models.generate_content(
                    model='gemma-4-31b-it', contents='left',
                    config=gtypes.GenerateContentConfig(temperature=0, max_output_tokens=4),
                )
                working.append(k)
                print(f"  Key {i:2d}: OK", flush=True)
                break
            except Exception as e:
                err = str(e)
                if '403' in err or '400' in err or 'quota' in err.lower() or '429' in err:
                    break
                time.sleep(2)
        time.sleep(0.3)
    return working


def query_llm(prompt, clients, key_last_call, gap, retries=3):
    for attempt in range(retries * len(clients)):
        now = time.time()
        idx = max(range(len(clients)), key=lambda i: now - key_last_call[i])
        elapsed = now - key_last_call[idx]
        if elapsed < gap:
            time.sleep(gap - elapsed)
        key_last_call[idx] = time.time()
        try:
            resp = clients[idx].models.generate_content(
                model='gemma-4-31b-it', contents=prompt,
                config=gtypes.GenerateContentConfig(temperature=0.0, max_output_tokens=8),
            )
            text = resp.text.strip() if resp.text else ''
            m = re.search(r'\b([1-9]|10)\b', text)
            if m:
                return int(m.group(1)) - 1
            return None
        except Exception as e:
            err = str(e)
            wait = 5.0 if ('quota' in err.lower() or '429' in err) else 2.0
            time.sleep(wait)
    return None


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Inference cache ──
    cache = build_or_load_inference_cache()
    print(f"Inference cache: {len(cache)} windows", flush=True)

    # Top-conf and min-ADE summary statistics
    top_ades = [v['top_ade'] for v in cache.values()]
    min_ades = [v['min_ade'] for v in cache.values()]
    print(f"  Mean top-conf ADE: {np.mean(top_ades):.3f} m", flush=True)
    print(f"  Mean min-ADE:      {np.mean(min_ades):.3f} m", flush=True)

    # ── 2. Load annotations ──
    df = pd.read_csv(ANNOT_CSV).dropna(subset=['Instruction'])
    df['scene_num'] = df['Scene Number'].astype(float).astype(int)
    df['ann_type']  = df['Instruction Type'].fillna('?')

    # Build set of scenes with windows
    val_scene_nums = set(sc for (sc, _) in cache.keys())
    df_val = df[df['scene_num'].isin(val_scene_nums)].copy()
    annotated_scenes = set(df_val['scene_num'])
    print(f"\n  Val scenes with windows: {len(val_scene_nums)}", flush=True)
    print(f"  Val scenes with annotations: {len(annotated_scenes)}", flush=True)
    print(f"  Total annotations available: {len(df_val)}", flush=True)

    # ── 3. Resume from existing JSONL ──
    seen_keys = set()
    if os.path.exists(RESULTS_JSONL):
        with open(RESULTS_JSONL) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    seen_keys.add((rec['scene'], rec['pkl'], rec['ann_id']))
                except Exception:
                    continue
        print(f"  Resuming: {len(seen_keys)} (window, annotation) pairs already done.", flush=True)

    # ── 4. Pre-flight Gemini keys ──
    print("", flush=True)
    working_keys = find_working_keys(KEYS_FILE)
    if not working_keys:
        print("ERROR: no working keys.", flush=True)
        return
    clients = [genai.Client(api_key=k) for k in working_keys]
    key_last_call = [0.0] * len(clients)
    print(f"  Using {len(clients)} working keys.\n", flush=True)

    # ── 5. Build the work queue: (sc, pkl, ann_record) ──
    # Each window paired with up to MAX_ANN_PER_WINDOW annotations.
    work = []
    for (sc, pkl), wd in cache.items():
        anns = annotations_for_scene(df_val, sc, MAX_ANN_PER_WINDOW)
        for ann in anns:
            ann_id = f"{ann['ann_type']}_{int(ann.name) if hasattr(ann, 'name') else 0}_{hash(ann['Instruction']) & 0xFFFFFF}"
            work.append((sc, pkl, ann, ann_id))
    print(f"  Work queue: {len(work)} (window, annotation) pairs (max {MAX_ANN_PER_WINDOW}/window)\n", flush=True)

    # ── 6. Process each pair sequentially, append results to JSONL ──
    print(f"  Estimated time @ ~10s/call sequential: {len(work) * 10 / 60:.0f} min\n", flush=True)
    f_out = open(RESULTS_JSONL, 'a')
    t0 = time.time()
    processed = 0
    skipped = 0
    for sc, pkl, ann, ann_id in work:
        key = (sc, pkl, ann_id)
        if key in seen_keys:
            skipped += 1
            continue
        wd = cache[(sc, pkl)]
        descs = [describe_trajectory(t) for t in wd['trajs']]
        prompt = make_prompt(str(ann['Instruction']).strip(), descs)
        picked = query_llm(prompt, clients, key_last_call, gap=CALL_GAP)
        if picked is None:
            picked = wd['top_k']        # fallback
            llm_ade = wd['top_ade']
            fb = True
        else:
            llm_ade = float(wd['dists'][picked])
            fb = False
        rec = {
            'scene': sc, 'pkl': pkl, 'ann_id': ann_id,
            'ann_type': ann['ann_type'], 'instruction': str(ann['Instruction']).strip(),
            'picked_idx': int(picked), 'fallback': fb,
            'llm_ade': llm_ade,
            'top_ade': wd['top_ade'], 'min_ade': wd['min_ade'],
        }
        f_out.write(json.dumps(rec) + '\n')
        f_out.flush()
        processed += 1

        if processed % 25 == 0 or processed == 1:
            elapsed = time.time() - t0
            rate = processed / max(1, elapsed)
            remaining = len(work) - processed - skipped
            eta_min = (remaining / rate) / 60 if rate > 0 else 0
            print(f"  [{time.strftime('%H:%M:%S')}] {processed} done "
                  f"({skipped} skipped); rate {rate:.2f}/s; ETA {eta_min:.0f}min", flush=True)

    f_out.close()
    print(f"\n  Total processed this run: {processed} (plus {skipped} skipped from cache)", flush=True)

    # ── 7. Aggregate from JSONL ──
    print("\n" + "=" * 100, flush=True)
    print("FULL VAL-SPLIT LLM ZERO-SHOT SELECTOR -- SUMMARY", flush=True)
    print("=" * 100, flush=True)

    records = []
    with open(RESULTS_JSONL) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    df_r = pd.DataFrame(records)
    print(f"  Total (window, annotation) records: {len(df_r)}", flush=True)
    print(f"  Distinct windows covered:           {df_r[['scene','pkl']].drop_duplicates().shape[0]}", flush=True)

    # Per-record (annotation-level)
    valid = df_r[~df_r['fallback']]
    print(f"\n  -- Per-(window, annotation) --", flush=True)
    print(f"  Total annotations queried: {len(df_r)}", flush=True)
    print(f"  Valid LLM picks:           {len(valid)}/{len(df_r)} "
          f"({100*len(valid)/max(1,len(df_r)):.1f}%)", flush=True)
    print(f"  Mean top-conf ADE:         {df_r['top_ade'].mean():.3f} m", flush=True)
    print(f"  Mean LLM-picked ADE:       {df_r['llm_ade'].mean():.3f} m", flush=True)
    print(f"  Mean min-ADE (oracle):     {df_r['min_ade'].mean():.3f} m", flush=True)
    delta = df_r['top_ade'] - df_r['llm_ade']
    print(f"  Mean Delta (top - LLM):    {delta.mean():+.3f} m", flush=True)
    print(f"  Improvement (Delta > 0.5 m): {(delta > 0.5).sum()}/{len(df_r)} "
          f"({100*(delta>0.5).sum()/max(1,len(df_r)):.1f}%)", flush=True)
    print(f"  ~Same   (|Delta| <= 0.5 m): {(delta.abs() <= 0.5).sum()}/{len(df_r)}", flush=True)
    print(f"  Worse   (Delta < -0.5 m):   {(delta < -0.5).sum()}/{len(df_r)}", flush=True)

    # Per-window aggregations
    print(f"\n  -- Per-window (1 entry per ego pickle) --", flush=True)
    grp = df_r.groupby(['scene', 'pkl'])
    win_top  = grp['top_ade'].first()
    win_min  = grp['min_ade'].first()
    win_mean = grp['llm_ade'].mean()        # mean over annotations
    win_best = grp['llm_ade'].min()         # oracle annotation choice

    n_ann_windows = len(win_top)
    n_unann_windows = len(cache) - n_ann_windows
    # Unannotated windows fall back to top_ade
    unann_top = sum(v['top_ade'] for (sc, pkl), v in cache.items()
                    if not ((df_r['scene'] == sc) & (df_r['pkl'] == pkl)).any())

    # All-window aggregations
    all_top = np.array([v['top_ade'] for v in cache.values()])
    all_min = np.array([v['min_ade'] for v in cache.values()])

    # Mean-over-annotations on annotated windows; top_ade on unannotated
    pair_lookup = {(sc, pkl): (mean_, best_)
                   for (sc, pkl), mean_, best_ in zip(win_mean.index, win_mean.values, win_best.values)}
    win_llm_mean_full = []
    win_llm_best_full = []
    for (sc, pkl), v in cache.items():
        if (sc, pkl) in pair_lookup:
            mn, bs = pair_lookup[(sc, pkl)]
        else:
            mn, bs = v['top_ade'], v['top_ade']
        win_llm_mean_full.append(mn)
        win_llm_best_full.append(bs)
    win_llm_mean_full = np.array(win_llm_mean_full)
    win_llm_best_full = np.array(win_llm_best_full)

    print(f"  Total windows in val:              {len(cache)}", flush=True)
    print(f"  Windows with annotations:          {n_ann_windows}", flush=True)
    print(f"  Windows without annotations:       {n_unann_windows}", flush=True)
    print(f"\n  Mean top-conf ADE  (current model):     {all_top.mean():.3f} m", flush=True)
    print(f"  Mean LLM-mean ADE  (avg over anns):     {win_llm_mean_full.mean():.3f} m   "
          f"  <- realistic deployment", flush=True)
    print(f"  Mean LLM-best ADE  (oracle ann choice): {win_llm_best_full.mean():.3f} m   "
          f"  <- ceiling per window", flush=True)
    print(f"  Mean min-ADE       (oracle K=10):       {all_min.mean():.3f} m   "
          f"  <- absolute oracle", flush=True)

    # Delta vs current top-conf
    print(f"\n  -- Improvement vs current model (top-conf) --", flush=True)
    print(f"  Delta (LLM-mean):    {(all_top - win_llm_mean_full).mean():+.3f} m  "
          f"({100*(all_top-win_llm_mean_full).mean()/max(0.001, all_top.mean()):.1f}% relative)", flush=True)
    print(f"  Delta (LLM-best):    {(all_top - win_llm_best_full).mean():+.3f} m  "
          f"({100*(all_top-win_llm_best_full).mean()/max(0.001, all_top.mean()):.1f}% relative)", flush=True)
    print(f"  Delta (oracle K=10): {(all_top - all_min).mean():+.3f} m  (max possible)", flush=True)

    # ADE percentiles for context
    print(f"\n  -- ADE percentile breakdown (over {len(cache)} windows) --", flush=True)
    print(f"  {'p25':>7}  {'p50':>7}  {'p75':>7}  {'p90':>7}  {'p95':>7}", flush=True)
    for label, arr in [('top-conf', all_top), ('LLM-mean', win_llm_mean_full),
                        ('LLM-best', win_llm_best_full), ('oracle', all_min)]:
        print(f"  {label:>11}: " + "  ".join(f"{np.percentile(arr, p):>6.2f}m"
                                              for p in [25, 50, 75, 90, 95]), flush=True)

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
