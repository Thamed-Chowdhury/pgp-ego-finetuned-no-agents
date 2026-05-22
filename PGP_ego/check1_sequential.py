"""
Check 1 — Sequential intent → GT alignment.

Samples up to 150 annotated val scenes (always includes all 19 high-ADE scenes).
Classifies NL intent with Gemini gemma-4-31b-it sequentially, cycling through
working keys at ≤ 15 RPM (4 s between consecutive calls to the same key).

Usage:
  cd /teamspace/studios/this_studio
  python3 -u PGP_ego/check1_sequential.py
"""

import os, sys, json, pickle, time, random
import numpy as np
import pandas as pd
from collections import Counter
from nuscenes.eval.prediction.splits import create_splits_scenes, NUM_IN_TRAIN_VAL
import google.genai as genai
from google.genai import types as gtypes

sys.path.insert(0, 'PGP_ego')

# ── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT   = 'nuscenes_data'
PREPROC_DIR = 'pgp_ego_preprocessed'
ANNOT_CSV   = 'annotated_doscenes.csv'
KEYS_FILE   = 'PGP_ego/Gemini_keys.txt'
HIGH_ADE    = [44,297,298,285,165,56,67,45,68,211,220,292,58,42,284,27,172,124,154]
MAX_ANNOTATIONS = 200     # hard cap on total annotations classified
RPM         = 14           # stay slightly below 15 to be safe
CALL_GAP    = 60.0 / RPM   # ~4.3 s between calls on the same key

LABELS = ['straight','turn_left','turn_right','lane_change_left',
          'lane_change_right','slow_stop','merge']

PROMPT = (
    "Classify the PRIMARY driving intent of this autonomous vehicle instruction "
    "as exactly one of: straight / turn_left / turn_right / lane_change_left / "
    "lane_change_right / slow_stop / merge.\n"
    "Instruction: \"{text}\"\n"
    "Reply with only the label, nothing else."
)

# ── GT direction from ego-frame trajectory ───────────────────────────────────
def gt_direction(traj):
    dist = float(np.linalg.norm(np.diff(traj, axis=0), axis=-1).sum())
    if dist < 3.0:
        return 'slow_stop'
    x_f, y_f = float(traj[-1, 0]), float(traj[-1, 1])
    a = float(np.degrees(np.arctan2(x_f, y_f)))
    if a < -20: return 'turn_left'
    if a >  20: return 'turn_right'
    if a <  -8: return 'lane_change_left'
    if a >   8: return 'lane_change_right'
    return 'straight'

# ── Build val scene→GT direction map ─────────────────────────────────────────
print("Building scene→GT-direction map ...", flush=True)
tv_scenes = set(create_splits_scenes()['train'][:NUM_IN_TRAIN_VAL])
with open(f'{DATA_ROOT}/v1.0-trainval/sample.json') as f:
    samples = json.load(f)
with open(f'{DATA_ROOT}/v1.0-trainval/scene.json') as f:
    scenes_j = json.load(f)
sc_tok_to_name = {s['token']: s['name'] for s in scenes_j}
sample_to_sc   = {s['token']: sc_tok_to_name.get(s['scene_token'], '') for s in samples}

scene_gt = {}
for fname in sorted(os.listdir(PREPROC_DIR)):
    if not fname.endswith('.pickle'):
        continue
    tok = fname[4:].replace('.pickle', '')
    sc_name = sample_to_sc.get(tok, '')
    if sc_name not in tv_scenes:
        continue
    sc_num = int(sc_name.split('-')[1])
    with open(f'{PREPROC_DIR}/{fname}', 'rb') as f:
        data = pickle.load(f)
    traj = np.array(data['ground_truth']['traj'])
    scene_gt.setdefault(sc_num, []).append(gt_direction(traj))

# Majority-vote per scene
scene_gt_dir = {sc: Counter(dirs).most_common(1)[0][0] for sc, dirs in scene_gt.items()}
print(f"  {len(scene_gt_dir)} val scenes with ego pickles.", flush=True)

# ── Load annotations ──────────────────────────────────────────────────────────
df = pd.read_csv(ANNOT_CSV).dropna(subset=['Instruction'])
df['scene_num'] = df['Scene Number'].astype(float).astype(int)
df['ann_type']  = df['Instruction Type'].fillna('?')
df_val = df[df['scene_num'].isin(scene_gt_dir)]

# Select annotations: always include all high-ADE scene annotations first,
# then fill up to MAX_ANNOTATIONS with a random sample from other scenes.
all_ann_scenes = sorted(df_val['scene_num'].unique())
ha_in_val = [s for s in HIGH_ADE if s in set(all_ann_scenes)]
other_scenes = [s for s in all_ann_scenes if s not in set(HIGH_ADE)]

df_ha  = df_val[df_val['scene_num'].isin(ha_in_val)]
budget = max(0, MAX_ANNOTATIONS - len(df_ha))

random.seed(42)
random.shuffle(other_scenes)
df_other_rows = []
for sc in other_scenes:
    sc_rows = df_val[df_val['scene_num'] == sc]
    if len(df_other_rows) + len(sc_rows) > budget:
        # Take as many as fit
        df_other_rows.append(sc_rows.iloc[:budget - len(df_other_rows)])
        break
    df_other_rows.append(sc_rows)
    if sum(len(r) for r in df_other_rows) >= budget:
        break

df_other = pd.concat(df_other_rows) if df_other_rows else pd.DataFrame()
df_sel   = pd.concat([df_ha, df_other]).copy()

print(f"  High-ADE annotations: {len(df_ha)} from {len(ha_in_val)} scenes.", flush=True)
print(f"  Other annotations:    {len(df_other)} from additional scenes.", flush=True)
print(f"  Total to classify:    {len(df_sel)}", flush=True)

# ── Find working Gemini keys ──────────────────────────────────────────────────
print("\nFinding working Gemini keys ...", flush=True)
with open(KEYS_FILE) as f:
    all_keys = [k.strip() for k in f if k.strip()]

working_keys = []
for i, k in enumerate(all_keys):
    c = genai.Client(api_key=k)
    for _ in range(2):
        try:
            c.models.generate_content(
                model='gemma-4-31b-it', contents='left',
                config=gtypes.GenerateContentConfig(temperature=0, max_output_tokens=4),
            )
            working_keys.append(k)
            print(f"  Key {i:2d}: OK", flush=True)
            break
        except Exception as e:
            err = str(e)
            if '403' in err or '400' in err or 'quota' in err.lower() or '429' in err:
                print(f"  Key {i:2d}: skip ({err[:40]})", flush=True)
                break
            time.sleep(2)
    time.sleep(0.3)

if not working_keys:
    print("ERROR: No working Gemini keys found.", flush=True)
    sys.exit(1)

print(f"  {len(working_keys)} working key(s).\n", flush=True)

# Build clients
clients = [genai.Client(api_key=k) for k in working_keys]
key_last_call = [0.0] * len(clients)   # track last call time per key

def next_key_idx():
    """Return index of the key with the longest idle time (most rested)."""
    now = time.time()
    return max(range(len(clients)), key=lambda i: now - key_last_call[i])

def classify(text):
    prompt = PROMPT.format(text=text)
    for attempt in range(len(clients) * 3):
        idx = next_key_idx()
        # Enforce CALL_GAP for this key
        elapsed = time.time() - key_last_call[idx]
        if elapsed < CALL_GAP:
            time.sleep(CALL_GAP - elapsed)
        key_last_call[idx] = time.time()
        try:
            resp = clients[idx].models.generate_content(
                model='gemma-4-31b-it', contents=prompt,
                config=gtypes.GenerateContentConfig(temperature=0.0, max_output_tokens=16),
            )
            raw = resp.text.strip().lower().rstrip('.')
            for l in LABELS:
                if l in raw:
                    return l
            return 'unknown'
        except Exception as e:
            err = str(e)
            wait = 6.0 if ('quota' in err.lower() or '429' in err) else 3.0
            time.sleep(wait)
    return 'error'

# ── Run classification ────────────────────────────────────────────────────────
print(f"{'Scene':>10}  {'T':>2}  {'GT direction':>18}  {'NL annotation (truncated)':43}  {'Gemini':>18}  Match")
print("-" * 110)

results = []
for i, (_, row) in enumerate(df_sel.sort_values('scene_num').iterrows()):
    sc      = int(row['scene_num'])
    gt_d    = scene_gt_dir[sc]
    text    = str(row['Instruction']).strip()
    ann_t   = str(row.get('ann_type', '?'))
    label   = classify(text)
    is_match  = (label == gt_d)
    is_close  = is_match or \
                (gt_d == 'straight' and label in ('lane_change_left','lane_change_right')) or \
                (label == 'straight' and gt_d in ('lane_change_left','lane_change_right'))
    sym = '✓' if is_match else ('~' if is_close else ('?' if label in ('error','unknown') else '✗'))
    print(f"scene-{sc:04d}  {ann_t:>2}  {gt_d:>18}  {text[:43]:43}  {label:>18}  {sym}", flush=True)
    results.append({
        'scene': sc, 'ann_type': ann_t, 'gt': gt_d,
        'gemini': label, 'match': is_match, 'close': is_close,
        'is_high_ade': sc in set(HIGH_ADE), 'text': text,
    })

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY — Check 1: Gemini Intent → GT Direction alignment")
print("=" * 70)

valid  = [r for r in results if r['gemini'] not in ('error','unknown')]
errors = [r for r in results if r['gemini'] in ('error','unknown')]
match  = [r for r in valid if r['match']]
close  = [r for r in valid if r['close']]

print(f"  Total annotations classified: {len(results)}")
print(f"  Valid labels:     {len(valid)}/{len(results)}  ({100*len(valid)/max(1,len(results)):.1f}%)")
print(f"  Exact match:      {len(match)}/{len(valid)}  ({100*len(match)/max(1,len(valid)):.1f}%)")
print(f"  Close match:      {len(close)}/{len(valid)}  ({100*len(close)/max(1,len(valid)):.1f}%)")
print(f"  Mismatch:         {len(valid)-len(close)}/{len(valid)}  ({100*(len(valid)-len(close))/max(1,len(valid)):.1f}%)")
print(f"  Errors/unknown:   {len(errors)}")

print("\n  By GT direction:")
by_dir = {}
for r in valid:
    by_dir.setdefault(r['gt'], {'m': 0, 'c': 0, 't': 0})
    by_dir[r['gt']]['t'] += 1
    if r['match']: by_dir[r['gt']]['m'] += 1
    if r['close']: by_dir[r['gt']]['c'] += 1
print(f"  {'GT Direction':>22}  {'Match/Total':>12}  {'Exact%':>8}  {'Close%':>8}")
for d, v in sorted(by_dir.items()):
    print(f"  {d:>22}  {v['m']:>5}/{v['t']:<5}  {100*v['m']/max(1,v['t']):>7.1f}%  {100*v['c']/max(1,v['t']):>7.1f}%")

print("\n  High-ADE scenes (19 worst):")
ha_res = [r for r in results if r['is_high_ade']]
print(f"  {'Scene':>10}  {'GT':>18}  {'Gemini labels'}")
for sc in HIGH_ADE:
    sc_r = [r for r in ha_res if r['scene'] == sc]
    if not sc_r:
        print(f"  scene-{sc:04d}  (no annotation found)")
        continue
    gt_d = sc_r[0]['gt']
    labels_str = '  |  '.join(
        f"{r['gemini']} {'✓' if r['match'] else ('?' if r['gemini'] in ('error','unknown') else '✗')}"
        for r in sc_r
    )
    print(f"  scene-{sc:04d}  {gt_d:>18}  {labels_str}")

print("\nDone.")
