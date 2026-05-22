"""
Two-check signal validation for NL-conditioned ego trajectory selection.

Check 1 — Intent → GT alignment
  For each annotated scene in the train_val split, extract a coarse intent label
  from the NL annotation (via Gemini gemma-4-31b-it) and verify it matches the
  actual GT trajectory direction.

Check 2 — Does the correct mode exist in K=10?
  For each of the 19 high-ADE scenes, run inference and check whether any of the
  K=10 predicted trajectories achieves ADE < 2 m against GT.
  If yes → re-ranking can fix this. If no → we need a generation fix.

Usage:
  cd /teamspace/studios/this_studio
  python PGP_ego/check_signal.py \
    -c PGP_ego/configs/pgp_ego_gatx2_lvm_traversal.yml \
    -r nuscenes_data \
    -d pgp_ego_preprocessed \
    -w pgp_ego_output2/checkpoints/best.tar \
    -a annotated_doscenes.csv \
    -k PGP_ego/Gemini_keys.txt
"""

import argparse
import json
import os
import pickle
import sys
import time

os.environ.setdefault('RAY_memory_monitor_refresh_ms', '0')

import asyncio
import numpy as np
import pandas as pd
import torch
import yaml

import google.genai as genai
from google.genai import types as gtypes

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from nuscenes.eval.prediction.splits import create_splits_scenes, NUM_IN_TRAIN_VAL
from train_eval.initialization import initialize_prediction_model, initialize_dataset, get_specific_args
import train_eval.utils as u

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

HIGH_ADE_SCENES = [44, 297, 298, 285, 165, 56, 67, 45, 68, 211, 220, 292, 58, 42, 284, 27, 172, 124, 154]

INTENT_LABELS = ['straight', 'turn_left', 'turn_right', 'lane_change_left', 'lane_change_right', 'slow_stop', 'merge']

INTENT_PROMPT = (
    "Classify driving intent as one label only: "
    "straight / turn_left / turn_right / lane_change_left / lane_change_right / slow_stop / merge.\n"
    "Instruction: \"{text}\"\nReply with only the label, no other text."
)

# ── Gemini key rotation ──────────────────────────────────────────────────────

def _parse_label(raw: str) -> str:
    label = raw.strip().lower().rstrip('.').strip()
    if label in INTENT_LABELS:
        return label
    for il in INTENT_LABELS:
        if il in label:
            return il
    return 'unknown'


def _test_key_sync(key: str) -> bool:
    """Quick sync pre-flight: returns True if key can call gemma-4-31b-it."""
    probe = 'left / right / straight? -> left'
    c = genai.Client(api_key=key)
    for _ in range(2):
        try:
            c.models.generate_content(
                model='gemma-4-31b-it', contents=probe,
                config=gtypes.GenerateContentConfig(temperature=0.0, max_output_tokens=4),
            )
            return True
        except Exception as e:
            err = str(e)
            if '403' in err or '400' in err or 'quota' in err.lower() or '429' in err:
                return False
            time.sleep(1)  # 500/transient: retry once
    return False


async def _classify_key_texts(client, indexed_texts: list, rpm: int,
                                retries: int, stagger_s: float) -> dict:
    """Process (orig_idx, text) pairs for one key, throttled to rpm, with initial stagger."""
    await asyncio.sleep(stagger_s)
    delay = 60.0 / rpm
    results = {}
    for pos, (orig_idx, text) in enumerate(indexed_texts):
        if pos > 0:
            await asyncio.sleep(delay)
        prompt = INTENT_PROMPT.format(text=text)
        label = 'error'
        for attempt in range(retries):
            try:
                resp = await client.aio.models.generate_content(
                    model='gemma-4-31b-it', contents=prompt,
                    config=gtypes.GenerateContentConfig(temperature=0.0, max_output_tokens=16),
                )
                label = _parse_label(resp.text)
                break
            except Exception as e:
                err = str(e)
                wait = 5.0 if ('quota' in err.lower() or '429' in err) else 3.0
                if attempt < retries - 1:
                    await asyncio.sleep(wait)
        results[orig_idx] = label
    return results


def classify_batch(texts: list, keys_file: str, rpm: int = 15, retries: int = 3) -> list:
    """
    1. Pre-flight test all keys to find working ones.
    2. Distribute texts round-robin across working keys.
    3. Each key worker is staggered 0.5s apart at start, then throttled to rpm.
    """
    with open(keys_file) as f:
        all_keys = [k.strip() for k in f if k.strip()]

    print(f"[Gemini] Pre-flight testing {len(all_keys)} keys ...", flush=True)
    working_keys = []
    for i, k in enumerate(all_keys):
        ok = _test_key_sync(k)
        status = 'OK' if ok else 'skip'
        print(f"  key {i:2d}: {status}", flush=True)
        if ok:
            working_keys.append(k)
        time.sleep(0.3)

    if not working_keys:
        print("ERROR: No working API keys found.", flush=True)
        return ['error'] * len(texts)

    n_keys = len(working_keys)
    buckets = {k: [] for k in working_keys}
    for i, text in enumerate(texts):
        buckets[working_keys[i % n_keys]].append((i, text))

    per_key = max(len(v) for v in buckets.values())
    est_s = per_key * (60.0 / rpm)
    print(f"[Gemini] {n_keys} working keys, ~{per_key} annotations/key, "
          f"est. {est_s:.0f}s total.", flush=True)

    clients_map = {k: genai.Client(api_key=k) for k in working_keys}

    async def run_all():
        tasks = [
            _classify_key_texts(
                clients_map[k], bucket, rpm, retries,
                stagger_s=idx * 0.5,  # stagger start times to avoid simultaneous burst
            )
            for idx, (k, bucket) in enumerate(buckets.items())
        ]
        dicts = await asyncio.gather(*tasks)
        merged = {}
        for d in dicts:
            merged.update(d)
        return [merged[i] for i in range(len(texts))]

    return asyncio.run(run_all())


# ── GT direction classifier ──────────────────────────────────────────────────

def classify_gt_direction(traj):
    """
    traj: (12, 2) numpy array in ego-local frame.
      x = lateral  (negative = left,  positive = right)
      y = forward  (positive = ahead)

    Returns one of: straight / turn_left / turn_right /
                    lane_change_left / lane_change_right / slow_stop
    """
    diffs = np.diff(traj, axis=0)
    total_dist = float(np.linalg.norm(diffs, axis=-1).sum())

    if total_dist < 3.0:
        return 'slow_stop'

    x_f, y_f = float(traj[-1, 0]), float(traj[-1, 1])
    angle_deg = float(np.degrees(np.arctan2(x_f, y_f)))  # neg=left, pos=right

    if angle_deg < -20:
        return 'turn_left'
    if angle_deg > 20:
        return 'turn_right'
    if angle_deg < -8:
        return 'lane_change_left'
    if angle_deg > 8:
        return 'lane_change_right'
    return 'straight'


# ── Data helpers ─────────────────────────────────────────────────────────────

def build_val_scene_pickle_map(preproc_dir, data_root):
    """
    Returns dict: scene_num → list of (pickle_path, gt_traj_np).
    Only for pickles whose sample token belongs to a train_val split scene.

    The train_val split is the first NUM_IN_TRAIN_VAL scenes of the 'train' split
    (as defined by create_splits_scenes). We map sample_token → scene via sample.json
    and scene.json, then filter by whether the scene name is in the train_val set.
    """
    # Determine train_val scene names
    tv_scene_names = set(create_splits_scenes()['train'][:NUM_IN_TRAIN_VAL])

    # Build sample_token → scene_name from JSON (no full NuScenes load)
    with open(os.path.join(data_root, 'v1.0-trainval', 'sample.json')) as f:
        samples = json.load(f)
    with open(os.path.join(data_root, 'v1.0-trainval', 'scene.json')) as f:
        scenes = json.load(f)
    scene_token_to_name = {s['token']: s['name'] for s in scenes}
    sample_to_scene_name = {
        s['token']: scene_token_to_name.get(s['scene_token'], '')
        for s in samples
    }

    scene_map = {}
    for pkl_file in sorted(os.listdir(preproc_dir)):
        if not pkl_file.endswith('.pickle'):
            continue
        tok = pkl_file[4:].replace('.pickle', '')   # strip 'ego_'
        sc_name = sample_to_scene_name.get(tok, '')
        if sc_name not in tv_scene_names:
            continue
        sc_num = int(sc_name.split('-')[1])
        full_path = os.path.join(preproc_dir, pkl_file)
        with open(full_path, 'rb') as f:
            data = pickle.load(f)
        gt_traj = np.array(data['ground_truth']['traj'])  # (12, 2)
        scene_map.setdefault(sc_num, []).append((full_path, gt_traj))
    return scene_map


# ── Check 1 ──────────────────────────────────────────────────────────────────

def run_check1(preproc_dir, data_root, annot_csv, keys_file):
    print("\n" + "=" * 70)
    print("CHECK 1 — Intent → GT alignment")
    print("=" * 70)

    print("Building scene→pickle mapping for train_val split...")
    scene_pickle_map = build_val_scene_pickle_map(preproc_dir, data_root)
    print(f"  {sum(len(v) for v in scene_pickle_map.values())} val-split pickle files "
          f"across {len(scene_pickle_map)} scenes.")

    # Classify GT direction for each scene (take the first valid window if multiple)
    scene_gt_direction = {}
    for scene_num, entries in scene_pickle_map.items():
        trajs = [gt for _, gt in entries]
        directions = [classify_gt_direction(t) for t in trajs]
        # Majority vote if multiple windows; ties go to first
        from collections import Counter
        scene_gt_direction[scene_num] = Counter(directions).most_common(1)[0][0]

    # Load annotations — include all types (s/d/ds/sd and NaN rows)
    # NaN-type rows are unlabelled but often contain clear directional instructions
    df = pd.read_csv(annot_csv)
    df.columns = df.columns.str.strip()
    df['Scene Number'] = df['Scene Number'].astype(float)
    df_intent = df.dropna(subset=['Instruction']).copy()
    df_intent['scene_num'] = df_intent['Scene Number'].astype(int)
    df_intent['ann_type'] = df_intent['Instruction Type'].fillna('?')

    # Filter to annotated scenes that are also in the val split
    annotated_val_scenes = sorted(set(df_intent['scene_num'].values) & set(scene_gt_direction.keys()))
    print(f"  {len(annotated_val_scenes)} val scenes have annotations (all types).")

    # Collect all (scene_num, ann_type, gt_dir, text) tuples for batch classification
    rows_to_classify = []
    for scene_num in annotated_val_scenes:
        gt_dir = scene_gt_direction[scene_num]
        scene_df = df_intent[df_intent['scene_num'] == scene_num]
        for _, row in scene_df.iterrows():
            rows_to_classify.append({
                'scene': scene_num,
                'gt_direction': gt_dir,
                'ann_type': str(row.get('ann_type', '?')),
                'annotation': str(row['Instruction']).strip(),
            })

    print(f"\n  Classifying {len(rows_to_classify)} annotations in parallel via gemma-4-31b-it ...", flush=True)
    texts = [r['annotation'] for r in rows_to_classify]
    labels = classify_batch(texts, keys_file, rpm=15, retries=3)

    print(f"\n  {'Scene':>7}  {'T':>2}  {'GT':>18}  {'NL Annotation (truncated)':43}  {'Gemini Intent':>18}  {'Match?':>6}")
    print("  " + "-" * 112)

    total, match, mismatch, unknown = 0, 0, 0, 0
    results = []
    for row_info, gemini_label in zip(rows_to_classify, labels):
        scene_num = row_info['scene']
        gt_dir    = row_info['gt_direction']
        ann_type  = row_info['ann_type']
        text      = row_info['annotation']
        is_match = (gemini_label == gt_dir)
        is_close = (
            is_match or
            (gt_dir == 'straight' and gemini_label in ('lane_change_left', 'lane_change_right')) or
            (gemini_label == 'straight' and gt_dir in ('lane_change_left', 'lane_change_right'))
        )
        symbol = '✓' if is_match else ('~' if is_close else '✗')
        if gemini_label in ('error', 'unknown'):
            symbol = '?'
            unknown += 1
        elif is_match:
            match += 1
        else:
            mismatch += 1
        total += 1
        results.append({
            'scene': scene_num,
            'ann_type': ann_type,
            'gt_direction': gt_dir,
            'annotation': text,
            'gemini_intent': gemini_label,
            'exact_match': is_match,
            'close_match': is_close,
        })
        print(f"  scene-{scene_num:04d}  {ann_type:>2}  {gt_dir:>18}  {text[:43]:43}  {gemini_label:>18}  {symbol:>6}")

    print("\n  ── Summary ─────────────────────────────────────────────────────────────")
    print(f"  Total annotations (all types, val split):  {total}")
    print(f"  Exact match  (Gemini label == GT dir):  {match}/{total}  ({100*match/max(1,total):.1f}%)")
    close = sum(r['close_match'] for r in results)
    print(f"  Close match  (lane_change ≈ straight):  {close}/{total}  ({100*close/max(1,total):.1f}%)")
    print(f"  Mismatch:                               {mismatch}/{total}")
    print(f"  Errors/unknown:                         {unknown}/{total}")

    # Breakdown by GT direction
    from collections import defaultdict
    by_dir = defaultdict(lambda: {'match': 0, 'total': 0})
    for r in results:
        by_dir[r['gt_direction']]['total'] += 1
        if r['exact_match']:
            by_dir[r['gt_direction']]['match'] += 1
    print("\n  ── By GT direction ─────────────────────────────────────────────────────")
    print(f"  {'GT Direction':>22}  {'Match/Total':>12}  {'Accuracy':>9}")
    for d, v in sorted(by_dir.items()):
        acc = 100 * v['match'] / max(1, v['total'])
        print(f"  {d:>22}  {v['match']:>5}/{v['total']:<5}  {acc:>8.1f}%")

    # Show specifically the 19 high-ADE scenes
    print("\n  ── High-ADE scenes (intent annotations) ───────────────────────────────")
    print(f"  {'Scene':>10}  {'GT Dir':>18}  {'Annotations'}  {'Gemini Labels'}")
    for scene_num in HIGH_ADE_SCENES:
        if scene_num not in scene_gt_direction:
            print(f"  scene-{scene_num:04d}  (no val-split pickle found)")
            continue
        gt_dir = scene_gt_direction[scene_num]
        scene_results = [r for r in results if r['scene'] == scene_num]
        if not scene_results:
            print(f"  scene-{scene_num:04d}  {gt_dir:>18}  (no annotation in val split)")
            continue
        labels = [r['gemini_intent'] for r in scene_results]
        matches = ['✓' if r['exact_match'] else '✗' for r in scene_results]
        label_str = '  |  '.join(f"{l} {m}" for l, m in zip(labels, matches))
        print(f"  scene-{scene_num:04d}  {gt_dir:>18}  {label_str}")

    return results, scene_gt_direction, scene_pickle_map


# ── Check 2 ──────────────────────────────────────────────────────────────────

def run_check2(scene_pickle_map, cfg, checkpoint):
    print("\n" + "=" * 70)
    print("CHECK 2 — Does the correct mode exist in K=10?")
    print("=" * 70)
    print("For each high-ADE scene: does any trajectory in K=10 have ADE < 2 m?\n")

    print("Loading ego model...")
    model = initialize_prediction_model(
        cfg['encoder_type'], cfg['aggregator_type'], cfg['decoder_type'],
        cfg['encoder_args'], cfg['aggregator_args'], cfg['decoder_args'],
    ).float().to(device)
    model.eval()
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.aggregator.num_samples = 100
    model.decoder.num_samples = 100
    print(f"  Loaded from {checkpoint}\n")

    print(f"  {'Scene':>10}  {'#Windows':>9}  {'minADE_10':>10}  {'<2m?':>6}  {'best_k_ade':>10}  {'top_conf_ade':>13}")
    print("  " + "-" * 80)

    results2 = []
    for scene_num in HIGH_ADE_SCENES:
        entries = scene_pickle_map.get(scene_num, [])
        if not entries:
            print(f"  scene-{scene_num:04d}  (not found in val split preprocessed data)")
            results2.append({'scene': scene_num, 'found': False})
            continue

        # Run inference on all windows for this scene, take the window with worst top-conf ADE
        # (to match the reported high-ADE scenario)
        scene_min_ades = []
        scene_top_ades = []
        best_window = None
        best_window_top_ade = -1

        for pkl_path, gt_traj in entries:
            with open(pkl_path, 'rb') as f:
                raw = pickle.load(f)

            data_t = u.send_to_device(u.convert_double_to_float(u.convert2tensors(raw)))
            with torch.no_grad():
                preds = model(data_t['inputs'])

            trajs = preds['traj'][0].detach().cpu().numpy()   # [K, T, 2]
            probs = preds['probs'][0].detach().cpu().numpy()  # [K]
            gt = gt_traj  # (12, 2)

            dists = np.linalg.norm(trajs - gt[None], axis=-1).mean(axis=1)  # [K]
            min_ade = float(dists.min())
            top_k = int(np.argmax(probs))
            top_ade = float(dists[top_k])

            scene_min_ades.append(min_ade)
            scene_top_ades.append(top_ade)
            if top_ade > best_window_top_ade:
                best_window_top_ade = top_ade
                best_window = {
                    'trajs': trajs, 'probs': probs, 'gt': gt,
                    'min_ade': min_ade, 'top_ade': top_ade, 'dists': dists
                }

        bw = best_window
        min_ade_10 = float(bw['min_ade'])
        top_conf_ade = float(bw['top_ade'])
        has_good_mode = min_ade_10 < 2.0

        # Per-trajectory ADEs for worst window (sorted best→worst)
        per_k = sorted(bw['dists'].tolist())

        results2.append({
            'scene': scene_num,
            'found': True,
            'n_windows': len(entries),
            'min_ade_10': min_ade_10,
            'top_conf_ade': top_conf_ade,
            'has_good_mode': has_good_mode,
            'per_k_ade': per_k,
        })

        ok = '✓ YES' if has_good_mode else '✗  NO'
        print(f"  scene-{scene_num:04d}  {len(entries):>9}  {min_ade_10:>10.3f}  {ok:>6}  "
              f"{per_k[0]:>10.3f}  {top_conf_ade:>13.3f}")

    # Summary
    found = [r for r in results2 if r['found']]
    has_good = [r for r in found if r['has_good_mode']]
    print("\n  ── Summary ─────────────────────────────────────────────────────────────")
    print(f"  Scenes with val-split pickles:     {len(found)}/{len(HIGH_ADE_SCENES)}")
    print(f"  Scenes where minADE_10 < 2 m:     {len(has_good)}/{len(found)}  "
          f"({'re-ranking CAN fix these' if has_good else 'need generation fix'})")
    no_good = [r for r in found if not r['has_good_mode']]
    if no_good:
        print(f"  Scenes where no good mode exists:  {len(no_good)}")
        for r in no_good:
            print(f"    scene-{r['scene']:04d}  minADE={r['min_ade_10']:.3f} m  "
                  f"best trajectory: {r['per_k_ade'][0]:.3f} m")

    # Per-trajectory breakdown for all found scenes
    print("\n  ── Per-trajectory ADE (sorted best→worst, worst window) ────────────────")
    print(f"  {'Scene':>10}  " + "  ".join(f"k={i+1:02d}" for i in range(10)))
    print("  " + "-" * 80)
    for r in results2:
        if not r['found']:
            continue
        vals = r['per_k_ade'][:10]
        vals_str = "  ".join(f"{v:5.2f}" for v in vals)
        flag = " ← ✓ good mode exists" if r['has_good_mode'] else " ← ✗ no good mode"
        print(f"  scene-{r['scene']:04d}  {vals_str}{flag}")

    return results2


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",     required=True)
    parser.add_argument("-r", "--data_root",  required=True)
    parser.add_argument("-d", "--data_dir",   required=True)
    parser.add_argument("-w", "--checkpoint", required=True)
    parser.add_argument("-a", "--annot_csv",  required=True)
    parser.add_argument("-k", "--keys_file",  required=True)
    parser.add_argument("--check", choices=['1', '2', 'both'], default='both',
                        help="Which check to run (default: both)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    scene_pkl_map = None

    # ── Check 1
    if args.check in ('1', 'both'):
        res1, scene_gt_dir, scene_pkl_map = run_check1(
            args.data_dir, args.data_root, args.annot_csv, args.keys_file
        )

    # ── Check 2 — needs the model and the scene→pickle map
    if args.check in ('2', 'both'):
        if scene_pkl_map is None:
            print("Building scene→pickle mapping for train_val split...")
            scene_pkl_map = build_val_scene_pickle_map(args.data_dir, args.data_root)

        run_check2(scene_pkl_map, cfg, args.checkpoint)

    print("\nDone.")


if __name__ == '__main__':
    main()
