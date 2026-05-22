"""
Zero-shot LLM trajectory selector — establishes the ceiling for NL re-ranking.

For each scene's worst-top-confidence window:
  1. Run the trained ego PGP model to get K=10 predicted trajectories.
  2. Convert each trajectory into a one-line natural-language description
     (deterministic geometry → text: direction, distance, speed profile).
  3. For each annotation, ask gemma-4-31b-it to pick the trajectory whose
     description best matches the driver's intent.
  4. Compute ADE of the LLM-selected trajectory vs ground truth.
  5. Compare against the model's top-confidence ADE (LVM probability) and
     best-of-K (minADE_10) for the same window.

Output: per-annotation and per-scene metrics, plus overall improvement.

Usage (defaults to the 19 high-ADE scenes):
  cd /teamspace/studios/this_studio
  python3 -u PGP_ego/llm_zero_shot_selector.py
"""

import os, sys, json, pickle, time, random, re
import numpy as np
import pandas as pd
import torch
import yaml

os.environ.setdefault('RAY_memory_monitor_refresh_ms', '0')

sys.path.insert(0, 'PGP_ego')
from train_eval.initialization import initialize_prediction_model
import train_eval.utils as u
from nuscenes.eval.prediction.splits import create_splits_scenes, NUM_IN_TRAIN_VAL
import google.genai as genai
from google.genai import types as gtypes

# ── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT    = 'nuscenes_data'
PREPROC_DIR  = 'pgp_ego_preprocessed'
ANNOT_CSV    = 'annotated_doscenes.csv'
KEYS_FILE    = 'PGP_ego/Gemini_keys.txt'
CFG_FILE     = 'PGP_ego/configs/pgp_ego_gatx2_lvm_traversal.yml'
CHECKPOINT   = 'pgp_ego_output2/checkpoints/best.tar'
OUT_DIR      = 'pgp_ego_check1_output'

HIGH_ADE = [44, 297, 298, 285, 165, 56, 67, 45, 68, 211, 220,
            292, 58, 42, 284, 27, 172, 124, 154]

CALL_GAP = 60.0 / 14   # 14 RPM — slightly under 15 to be safe

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ── Trajectory → natural-language description ────────────────────────────────
def describe_trajectory(traj, t_per_step: float = 0.5) -> str:
    """
    traj: (12, 2) numpy array in local ego frame.
      x = lateral (positive = right), y = forward.
    Returns a deterministic one-line description.
    """
    diffs = np.diff(traj, axis=0)
    step_dists = np.linalg.norm(diffs, axis=-1)
    speeds = step_dists / t_per_step

    init_v  = float(speeds[:3].mean())
    final_v = float(speeds[-3:].mean())
    total   = float(step_dists.sum())

    x_f, y_f = float(traj[-1, 0]), float(traj[-1, 1])
    end_angle = float(np.degrees(np.arctan2(x_f, y_f)))   # neg = left, pos = right

    # Maximum lateral deviation (peak side excursion)
    max_lat = float(np.max(np.abs(traj[:, 0])))

    # Direction phrase
    if total < 3.0:
        direction = "barely moves (likely stops)"
    elif abs(end_angle) < 4:
        direction = "goes straight"
    elif abs(end_angle) < 10:
        direction = f"mostly straight with slight {'left' if end_angle < 0 else 'right'}ward drift"
    elif abs(end_angle) < 22:
        direction = f"curves {'left' if end_angle < 0 else 'right'} ~{abs(end_angle):.0f} deg"
    else:
        direction = f"turns {'left' if end_angle < 0 else 'right'} {abs(end_angle):.0f} deg"

    # Lateral hint (lane change vs in-lane)
    if abs(end_angle) < 10 and max_lat > 1.5:
        side = 'left' if traj[np.argmax(np.abs(traj[:, 0])), 0] < 0 else 'right'
        direction += f" with up to {max_lat:.1f}m {side}ward shift"

    # Speed phrase
    if max(init_v, final_v) < 0.8:
        speed = "stationary"
    elif abs(final_v - init_v) < 1.0:
        speed = f"~constant speed (~{init_v:.1f} m/s)"
    elif final_v < init_v - 1.0:
        speed = f"decelerates {init_v:.1f}->{final_v:.1f} m/s"
    else:
        speed = f"accelerates {init_v:.1f}->{final_v:.1f} m/s"

    return f"{direction}; travels {total:.0f}m; {speed}"


# ── Build val scene → ego pickle map ────────────────────────────────────────
def build_val_scene_pickle_map(preproc_dir: str, data_root: str) -> dict:
    tv_scenes = set(create_splits_scenes()['train'][:NUM_IN_TRAIN_VAL])
    trainval_dir = os.path.join(data_root, 'trainval') if os.path.exists(os.path.join(data_root, 'trainval')) else os.path.join(data_root, 'v1.0-trainval')
    with open(os.path.join(trainval_dir, 'sample.json')) as f:
        samples = json.load(f)
    with open(os.path.join(trainval_dir, 'scene.json')) as f:
        scenes = json.load(f)
    sc_tok_to_name = {s['token']: s['name'] for s in scenes}
    sample_to_sc   = {s['token']: sc_tok_to_name.get(s['scene_token'], '') for s in samples}

    out = {}
    for f in sorted(os.listdir(preproc_dir)):
        if not f.endswith('.pickle'):
            continue
        tok = f[4:].replace('.pickle', '')
        sc_name = sample_to_sc.get(tok, '')
        if sc_name not in tv_scenes:
            continue
        sc_num = int(sc_name.split('-')[1])
        full = os.path.join(preproc_dir, f)
        out.setdefault(sc_num, []).append(full)
    return out


# ── Run model inference on one window ────────────────────────────────────────
def run_inference(model, pkl_path: str):
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    data_t = u.send_to_device(u.convert_double_to_float(u.convert2tensors(raw)))
    with torch.no_grad():
        preds = model(data_t['inputs'])
    trajs = preds['traj'][0].detach().cpu().numpy()    # (K, T, 2)
    probs = preds['probs'][0].detach().cpu().numpy()   # (K,)
    gt    = data_t['ground_truth']['traj'][0].detach().cpu().numpy()  # (T, 2)
    return trajs, probs, gt


# ── Gemini key pool: pre-flight + key-rotation ──────────────────────────────
def find_working_keys(keys_file: str) -> list:
    with open(keys_file) as f:
        all_keys = [k.strip() for k in f if k.strip()]
    print(f"Pre-flight testing {len(all_keys)} Gemini keys ...", flush=True)
    working = []
    for i, k in enumerate(all_keys):
        c = genai.Client(api_key=k)
        for attempt in range(2):
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
                    print(f"  Key {i:2d}: skip ({err[:30]})", flush=True)
                    break
                time.sleep(2)
        time.sleep(0.3)
    return working


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


def query_llm(prompt: str, clients: list, key_last_call: list, gap: float, retries: int = 3) -> int:
    for attempt in range(retries * len(clients)):
        # Pick the most-rested key
        now = time.time()
        idx = max(range(len(clients)), key=lambda i: now - key_last_call[i])
        elapsed = now - key_last_call[idx]
        if elapsed < gap:
            time.sleep(gap - elapsed)
        key_last_call[idx] = time.time()
        try:
            resp = clients[idx].models.generate_content(
                model='gemma-4-31b-it',
                contents=prompt,
                config=gtypes.GenerateContentConfig(temperature=0.0, max_output_tokens=8),
            )
            text = resp.text.strip() if resp.text else ''
            m = re.search(r'\b([1-9]|10)\b', text)
            if m:
                return int(m.group(1)) - 1   # 0-indexed
            return None
        except Exception as e:
            err = str(e)
            wait = 5.0 if ('quota' in err.lower() or '429' in err) else 2.0
            time.sleep(wait)
    return None


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load model
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

    # Build scene→pickle map
    print("Building val scene→pickle map ...", flush=True)
    scene_map = build_val_scene_pickle_map(PREPROC_DIR, DATA_ROOT)

    # Load annotations
    df = pd.read_csv(ANNOT_CSV).dropna(subset=['Instruction'])
    df['scene_num'] = df['Scene Number'].astype(float).astype(int)
    df['ann_type']  = df['Instruction Type'].fillna('?')

    # ── Run inference + describe trajectories for each high-ADE scene ──
    print("\nRunning inference on 19 high-ADE scenes ...", flush=True)
    scene_data = {}
    for sc in HIGH_ADE:
        entries = scene_map.get(sc, [])
        if not entries:
            print(f"  scene-{sc:04d}: no pickle", flush=True)
            continue

        windows = []
        for pkl in entries:
            trajs, probs, gt = run_inference(model, pkl)
            dists = np.linalg.norm(trajs - gt[None], axis=-1).mean(axis=1)  # (K,)
            top_k = int(np.argmax(probs))
            windows.append({
                'pkl': pkl, 'trajs': trajs, 'probs': probs, 'gt': gt,
                'dists': dists, 'top_k': top_k, 'top_ade': float(dists[top_k]),
            })
        # Use the worst top-conf window (matches the high-ADE evaluation)
        worst = max(windows, key=lambda w: w['top_ade'])
        scene_data[sc] = worst

        descs = [describe_trajectory(t) for t in worst['trajs']]
        # Print top-conf and best-of-K for sanity
        bk = int(np.argmin(worst['dists']))
        print(f"  scene-{sc:04d}: top_conf_ADE={worst['top_ade']:.2f}m  "
              f"min_ADE={worst['dists'].min():.2f}m  "
              f"GT_endpoint=({worst['gt'][-1,0]:.1f},{worst['gt'][-1,1]:.1f})", flush=True)
        for i, d in enumerate(descs):
            mark = ''
            if i == worst['top_k']: mark += ' [top-conf]'
            if i == bk: mark += ' [best-of-K]'
            print(f"    {i+1:>2}. ADE={worst['dists'][i]:.2f}m  {d}{mark}", flush=True)

    # ── Pre-flight Gemini keys ──
    print("\nFinding working Gemini keys ...", flush=True)
    working_keys = find_working_keys(KEYS_FILE)
    if not working_keys:
        print("ERROR: no working keys.", flush=True)
        return
    clients = [genai.Client(api_key=k) for k in working_keys]
    key_last_call = [0.0] * len(clients)
    print(f"  Using {len(clients)} working keys.", flush=True)

    # ── Run LLM selector for each annotation of each high-ADE scene ──
    print("\n" + "=" * 100, flush=True)
    print("ZERO-SHOT LLM SELECTOR — Gemini gemma-4-31b-it", flush=True)
    print("=" * 100, flush=True)
    print(f"{'Scene':>10}  {'T':>2}  {'Annotation (truncated)':47}  "
          f"{'Pick':>5}  {'LLM ADE':>8}  {'Top ADE':>8}  {'minADE':>7}  "
          f"{'Δ vs Top':>9}", flush=True)
    print("-" * 110, flush=True)

    results = []
    for sc in HIGH_ADE:
        if sc not in scene_data:
            continue
        wd = scene_data[sc]
        descs = [describe_trajectory(t) for t in wd['trajs']]
        scene_anns = df[df['scene_num'] == sc]
        for _, row in scene_anns.iterrows():
            text = str(row['Instruction']).strip()
            ann_t = str(row['ann_type'])
            prompt = make_prompt(text, descs)
            picked = query_llm(prompt, clients, key_last_call, gap=CALL_GAP)
            if picked is None:
                pick_str = 'ERR'
                llm_ade = wd['top_ade']  # fallback
                fallback = True
            else:
                pick_str = str(picked + 1)
                llm_ade = float(wd['dists'][picked])
                fallback = False
            min_ade = float(wd['dists'].min())
            delta = wd['top_ade'] - llm_ade
            improvement = 'v' if delta > 0.5 else ('=' if abs(delta) <= 0.5 else '^')
            print(f"  scene-{sc:04d}  {ann_t:>2}  {text[:47]:47}  "
                  f"{pick_str:>5}  {llm_ade:>7.2f}m  {wd['top_ade']:>7.2f}m  "
                  f"{min_ade:>6.2f}m  {improvement} {delta:+5.2f}m", flush=True)
            results.append({
                'scene': sc, 'ann_type': ann_t, 'text': text,
                'picked_idx': picked, 'fallback': fallback,
                'llm_ade': llm_ade, 'top_ade': wd['top_ade'],
                'min_ade': min_ade, 'delta': delta,
            })

    # ── Aggregate ──
    print("\n" + "=" * 100, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 100, flush=True)

    valid = [r for r in results if not r['fallback']]
    if not valid:
        print("No valid LLM picks.", flush=True)
        return

    llm_ades = np.array([r['llm_ade']  for r in valid])
    top_ades = np.array([r['top_ade']  for r in valid])
    min_ades = np.array([r['min_ade']  for r in valid])
    deltas   = top_ades - llm_ades

    print(f"  Annotations queried:        {len(results)}")
    print(f"  Valid LLM picks:            {len(valid)}/{len(results)}")
    print(f"  Mean ADE (top-confidence):  {top_ades.mean():.3f} m  (current model)")
    print(f"  Mean ADE (LLM-selected):    {llm_ades.mean():.3f} m  (zero-shot ceiling)")
    print(f"  Mean ADE (best-of-K):       {min_ades.mean():.3f} m  (oracle)")
    print(f"  Mean Δ (top - LLM):         {deltas.mean():+.3f} m   "
          f"({'better' if deltas.mean() > 0 else 'worse'})")

    improved = (deltas > 0.5).sum()
    same     = (np.abs(deltas) <= 0.5).sum()
    worse    = (deltas < -0.5).sum()
    print(f"  Improvement (Δ > 0.5 m):    {improved}/{len(valid)}  ({100*improved/len(valid):.1f}%)")
    print(f"  ~Same   (|Δ| ≤ 0.5 m):      {same}/{len(valid)}  ({100*same/len(valid):.1f}%)")
    print(f"  Worse   (Δ < -0.5 m):       {worse}/{len(valid)}  ({100*worse/len(valid):.1f}%)")

    # Per-scene best (the most optimistic re-rank — pick the annotation that
    # leads the LLM closest to GT)
    print(f"\n  Per-scene BEST annotation (oracle annotation choice):")
    print(f"  {'Scene':>10}  {'top':>7}  {'LLM_best':>8}  {'min':>7}  {'Δ best vs top':>13}")
    by_scene = {}
    for r in valid:
        by_scene.setdefault(r['scene'], []).append(r)
    scene_best = []
    for sc, rs in sorted(by_scene.items()):
        best_r = min(rs, key=lambda r: r['llm_ade'])
        d = best_r['top_ade'] - best_r['llm_ade']
        print(f"  scene-{sc:04d}  {best_r['top_ade']:>6.2f}m  "
              f"{best_r['llm_ade']:>7.2f}m  {best_r['min_ade']:>6.2f}m  "
              f"{d:+10.2f}m")
        scene_best.append({'scene': sc, 'top': best_r['top_ade'],
                           'llm_best': best_r['llm_ade'],
                           'min': best_r['min_ade']})

    sb = pd.DataFrame(scene_best)
    print(f"\n  Across {len(sb)} scenes (best annotation per scene):")
    print(f"    Mean top-conf ADE:       {sb['top'].mean():.3f} m")
    print(f"    Mean LLM-best ADE:       {sb['llm_best'].mean():.3f} m")
    print(f"    Mean best-of-K (oracle): {sb['min'].mean():.3f} m")

    # Save full results to CSV
    csv_path = os.path.join(OUT_DIR, 'llm_zero_shot_results.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\nFull per-annotation results saved to {csv_path}")
    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
