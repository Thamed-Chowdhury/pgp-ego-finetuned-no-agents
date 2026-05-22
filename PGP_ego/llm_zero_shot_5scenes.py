"""
Sanity-check LLM zero-shot reranking on 5 random scenes from the nuScenes
val (train_val) split that have driver annotations.

Mirrors llm_zero_shot_selector.py but:
  - uses absolute paths for this local setup
  - selects 5 random annotated val scenes instead of HIGH_ADE
"""

import os, sys, json, pickle, time, random, re
import numpy as np
import pandas as pd
import torch
import yaml

os.environ.setdefault('RAY_memory_monitor_refresh_ms', '0')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from train_eval.initialization import initialize_prediction_model
import train_eval.utils as u
import google.genai as genai
from google.genai import types as gtypes

from llm_zero_shot_selector import (
    describe_trajectory, build_val_scene_pickle_map, run_inference,
    find_working_keys, make_prompt, query_llm,
)

# Absolute paths for this machine
DATA_ROOT   = r'D:\DriveX_PGP\nuScenes_data'
PREPROC_DIR = r'D:\DriveX_PGP\pgp-ego-prediction\pgp_ego_preprocessed'
ANNOT_CSV   = r'D:\DriveX_PGP\doScenes-VLM-Planning-main\data\doScenes\annotated_doscenes.csv'
KEYS_FILE   = r'D:\DriveX_PGP\Gemini_keys.txt'
CFG_FILE    = os.path.join(HERE, 'configs', 'pgp_ego_gatx2_lvm_traversal.yml')
CHECKPOINT  = r'D:\DriveX_PGP\pgp-ego-prediction\pgp_ego_output2\checkpoints\best.tar'
OUT_DIR     = r'D:\DriveX_PGP\pgp-ego-prediction\pgp_ego_5scene_check'

NUM_SCENES = 5
SEED = 42
CALL_GAP = 60.0 / 14

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {device}", flush=True)

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

    print("Building val scene -> pickle map ...", flush=True)
    scene_map = build_val_scene_pickle_map(PREPROC_DIR, DATA_ROOT)
    print(f"  Val scenes with windows: {len(scene_map)}", flush=True)

    df = pd.read_csv(ANNOT_CSV).dropna(subset=['Instruction'])
    df['scene_num'] = df['Scene Number'].astype(float).astype(int)
    df['ann_type']  = df['Instruction Type'].fillna('?')

    annotated_val_scenes = sorted(set(df['scene_num']) & set(scene_map.keys()))
    print(f"  Val scenes with annotations: {len(annotated_val_scenes)}", flush=True)

    rng = random.Random(SEED)
    chosen = sorted(rng.sample(annotated_val_scenes, NUM_SCENES))
    print(f"  Randomly chose 5 scenes (seed={SEED}): {chosen}\n", flush=True)

    print("Running inference ...", flush=True)
    scene_data = {}
    for sc in chosen:
        entries = scene_map.get(sc, [])
        if not entries:
            print(f"  scene-{sc:04d}: no pickle", flush=True)
            continue
        windows = []
        for pkl in entries:
            trajs, probs, gt = run_inference(model, pkl)
            dists = np.linalg.norm(trajs - gt[None], axis=-1).mean(axis=1)
            top_k = int(np.argmax(probs))
            windows.append({
                'pkl': pkl, 'trajs': trajs, 'probs': probs, 'gt': gt,
                'dists': dists, 'top_k': top_k, 'top_ade': float(dists[top_k]),
            })
        worst = max(windows, key=lambda w: w['top_ade'])
        scene_data[sc] = worst
        descs = [describe_trajectory(t) for t in worst['trajs']]
        bk = int(np.argmin(worst['dists']))
        print(f"  scene-{sc:04d}: top_conf_ADE={worst['top_ade']:.2f}m  "
              f"min_ADE={worst['dists'].min():.2f}m  "
              f"GT_endpoint=({worst['gt'][-1,0]:.1f},{worst['gt'][-1,1]:.1f})", flush=True)
        for i, d in enumerate(descs):
            mark = ''
            if i == worst['top_k']: mark += ' [top-conf]'
            if i == bk: mark += ' [best-of-K]'
            print(f"    {i+1:>2}. ADE={worst['dists'][i]:.2f}m  {d}{mark}", flush=True)

    print("\nFinding working Gemini keys ...", flush=True)
    working_keys = find_working_keys(KEYS_FILE)
    if not working_keys:
        print("ERROR: no working keys.", flush=True)
        return
    clients = [genai.Client(api_key=k) for k in working_keys]
    key_last_call = [0.0] * len(clients)
    print(f"  Using {len(clients)} working keys.", flush=True)

    print("\n" + "=" * 100, flush=True)
    print("ZERO-SHOT LLM SELECTOR -- 5 random val scenes", flush=True)
    print("=" * 100, flush=True)
    print(f"{'Scene':>10}  {'T':>2}  {'Annotation (truncated)':47}  "
          f"{'Pick':>5}  {'LLM ADE':>8}  {'Top ADE':>8}  {'minADE':>7}  "
          f"{'Delta vs Top':>13}", flush=True)
    print("-" * 110, flush=True)

    results = []
    for sc in chosen:
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
                pick_str, llm_ade, fallback = 'ERR', wd['top_ade'], True
            else:
                pick_str = str(picked + 1)
                llm_ade = float(wd['dists'][picked])
                fallback = False
            min_ade = float(wd['dists'].min())
            delta = wd['top_ade'] - llm_ade
            arrow = 'D' if delta > 0.5 else ('=' if abs(delta) <= 0.5 else 'U')
            print(f"  scene-{sc:04d}  {ann_t:>2}  {text[:47]:47}  "
                  f"{pick_str:>5}  {llm_ade:>7.2f}m  {wd['top_ade']:>7.2f}m  "
                  f"{min_ade:>6.2f}m  {arrow} {delta:+5.2f}m", flush=True)
            results.append({
                'scene': sc, 'ann_type': ann_t, 'text': text,
                'picked_idx': picked, 'fallback': fallback,
                'llm_ade': llm_ade, 'top_ade': wd['top_ade'],
                'min_ade': min_ade, 'delta': delta,
            })

    print("\n" + "=" * 100, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 100, flush=True)

    valid = [r for r in results if not r['fallback']]
    print(f"  Annotations queried:        {len(results)}")
    print(f"  Valid LLM picks:            {len(valid)}/{len(results)}")
    if valid:
        llm_ades = np.array([r['llm_ade']  for r in valid])
        top_ades = np.array([r['top_ade']  for r in valid])
        min_ades = np.array([r['min_ade']  for r in valid])
        deltas   = top_ades - llm_ades
        print(f"  Mean ADE (top-confidence):  {top_ades.mean():.3f} m")
        print(f"  Mean ADE (LLM-selected):    {llm_ades.mean():.3f} m")
        print(f"  Mean ADE (best-of-K):       {min_ades.mean():.3f} m")
        print(f"  Mean Delta (top - LLM):     {deltas.mean():+.3f} m")

    csv_path = os.path.join(OUT_DIR, 'llm_5scene_results.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\nFull per-annotation results saved to {csv_path}")
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
