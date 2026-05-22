"""
Diagnose why LLM-picked trajectories perform worse than top-confidence.

Loads:
  - documentation/full_val_llm_results.jsonl  (832 records)
  - documentation/inference_cache.pkl         (373 windows w/ trajs, probs, gt, dists)

Reports failure modes:
  1. Picked-index distribution (is the LLM biased toward certain slots?)
  2. Confidence rank of LLM pick (does it pick low-prob trajectories?)
  3. Direction agreement: does LLM pick a trajectory whose end-direction
     matches the annotation's stated direction?
  4. Per-annotation-type breakdown (s vs d vs sd vs ds vs ?)
  5. Worst cases: 10 examples where LLM picked something far from GT.
"""
import os, json, pickle
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DOC  = os.path.join(HERE, 'documentation')
JSONL = os.path.join(DOC, 'full_val_llm_results.jsonl')
CACHE = os.path.join(DOC, 'inference_cache.pkl')

# Load
records = []
with open(JSONL) as f:
    for line in f:
        try:
            records.append(json.loads(line))
        except Exception:
            pass
df = pd.DataFrame(records)
print(f"Records loaded: {len(df)}")

with open(CACHE, 'rb') as f:
    cache = pickle.load(f)
print(f"Cache windows:  {len(cache)}")

# Build (scene,pkl) -> {trajs, probs, gt, dists, top_k, top_ade, min_ade}
def lookup(scene, pkl):
    return cache.get((scene, pkl))

# ---- 1. Picked-index distribution -------------------------------------------
print("\n" + "=" * 80)
print("1. PICKED-INDEX DISTRIBUTION (0-9)")
print("=" * 80)
valid = df[~df['fallback']].copy()
print(f"Valid (non-fallback) picks: {len(valid)}/{len(df)}")
counts = valid['picked_idx'].value_counts().sort_index()
total = len(valid)
print(f"\n  idx  count  pct")
for i in range(10):
    c = counts.get(i, 0)
    bar = '#' * int(60 * c / total)
    print(f"   {i}: {c:4d}  {100*c/total:5.1f}%  {bar}")

# Compare to model's top_k distribution
top_k_per_record = []
for _, r in df.iterrows():
    wd = lookup(r['scene'], r['pkl'])
    if wd is not None:
        top_k_per_record.append(wd['top_k'])
df['top_k'] = top_k_per_record
print(f"\nModel's top-k distribution (most-confident slot per window):")
top_k_counts = df['top_k'].value_counts().sort_index()
for i in range(10):
    c = top_k_counts.get(i, 0)
    print(f"   {i}: {c:4d}  {100*c/len(df):5.1f}%")

# ---- 2. Does LLM pick low-probability trajectories? -------------------------
print("\n" + "=" * 80)
print("2. CONFIDENCE-RANK OF LLM PICK (1 = highest prob, 10 = lowest)")
print("=" * 80)
ranks = []
prob_pickeds = []
prob_topks = []
for _, r in valid.iterrows():
    wd = lookup(r['scene'], r['pkl'])
    if wd is None: continue
    probs = wd['probs']
    pi = int(r['picked_idx'])
    rank = int((np.argsort(-probs) == pi).nonzero()[0][0]) + 1
    ranks.append(rank)
    prob_pickeds.append(float(probs[pi]))
    prob_topks.append(float(probs.max()))
ranks = np.array(ranks)
print(f"  Mean rank of LLM pick:         {ranks.mean():.2f}  (uniform = 5.5)")
print(f"  Median rank:                   {np.median(ranks):.0f}")
print(f"  % picks with rank 1 (top-k):   {100*(ranks==1).mean():.1f}%")
print(f"  % picks with rank in top-3:    {100*(ranks<=3).mean():.1f}%")
print(f"  % picks with rank in bot-3:    {100*(ranks>=8).mean():.1f}%")
print(f"  Mean prob of LLM pick:         {np.mean(prob_pickeds):.3f}")
print(f"  Mean prob of model top-k:      {np.mean(prob_topks):.3f}")

# ---- 3. ADE of LLM pick vs random vs top vs min -----------------------------
print("\n" + "=" * 80)
print("3. ADE COMPARISON (mean over valid picks)")
print("=" * 80)
random_ades = []
for _, r in valid.iterrows():
    wd = lookup(r['scene'], r['pkl'])
    if wd is None: continue
    random_ades.append(float(wd['dists'].mean()))   # mean over K
print(f"  LLM-picked:           {valid['llm_ade'].mean():.3f} m")
print(f"  Top-conf:             {valid['top_ade'].mean():.3f} m")
print(f"  Random / Mean-of-K:   {np.mean(random_ades):.3f} m")
print(f"  Min-ADE (oracle):     {valid['min_ade'].mean():.3f} m")
print(f"  Worst-of-K (max):     ", end='')
worst_ades = []
for _, r in valid.iterrows():
    wd = lookup(r['scene'], r['pkl'])
    if wd is None: continue
    worst_ades.append(float(wd['dists'].max()))
print(f"{np.mean(worst_ades):.3f} m")

# ---- 4. Direction agreement -------------------------------------------------
print("\n" + "=" * 80)
print("4. DIRECTION AGREEMENT")
print("=" * 80)
def end_angle_deg(traj):
    x, y = float(traj[-1, 0]), float(traj[-1, 1])
    return float(np.degrees(np.arctan2(x, y)))   # neg = left, pos = right

def cls_dir(angle):
    if abs(angle) < 4:  return 'straight'
    if abs(angle) < 22: return 'left' if angle < 0 else 'right'
    return 'turn-left' if angle < 0 else 'turn-right'

# Direction of GT, top-pick, LLM-pick per record
agree_top = 0; agree_llm = 0; total = 0
gt_dirs = []
for _, r in valid.iterrows():
    wd = lookup(r['scene'], r['pkl'])
    if wd is None: continue
    gt_d   = cls_dir(end_angle_deg(wd['gt']))
    top_d  = cls_dir(end_angle_deg(wd['trajs'][int(wd['top_k'])]))
    llm_d  = cls_dir(end_angle_deg(wd['trajs'][int(r['picked_idx'])]))
    gt_dirs.append(gt_d)
    if top_d == gt_d: agree_top += 1
    if llm_d == gt_d: agree_llm += 1
    total += 1
print(f"  Records: {total}")
print(f"  Top-conf direction == GT direction: {agree_top}/{total}  ({100*agree_top/total:.1f}%)")
print(f"  LLM     direction == GT direction:  {agree_llm}/{total}  ({100*agree_llm/total:.1f}%)")

print(f"\n  GT direction distribution:")
gt_dir_counts = pd.Series(gt_dirs).value_counts()
for d, c in gt_dir_counts.items():
    print(f"    {d:>10}: {c:4d}  ({100*c/total:5.1f}%)")

# ---- 5. Per-annotation-type breakdown ----------------------------------------
print("\n" + "=" * 80)
print("5. PER-ANNOTATION-TYPE BREAKDOWN")
print("=" * 80)
print(f"  {'type':>5}  {'count':>5}  {'top_ADE':>8}  {'LLM_ADE':>8}  {'min_ADE':>8}  {'Delta':>7}")
for ann_t, sub in df.groupby('ann_type'):
    delta = sub['top_ade'].mean() - sub['llm_ade'].mean()
    print(f"  {ann_t:>5}  {len(sub):>5}  {sub['top_ade'].mean():>7.2f}m "
          f" {sub['llm_ade'].mean():>7.2f}m  {sub['min_ade'].mean():>7.2f}m "
          f" {delta:+7.2f}m")

# ---- 6. Where does the LLM go wrong most? -----------------------------------
print("\n" + "=" * 80)
print("6. 10 WORST CASES (LLM picked far worse than top-conf)")
print("=" * 80)
df['delta'] = df['top_ade'] - df['llm_ade']
worst = df.nsmallest(10, 'delta')
for i, r in enumerate(worst.itertuples(), 1):
    wd = lookup(r.scene, r.pkl)
    if wd is None: continue
    gt_a   = end_angle_deg(wd['gt'])
    top_a  = end_angle_deg(wd['trajs'][int(wd['top_k'])])
    llm_a  = end_angle_deg(wd['trajs'][int(r.picked_idx)])
    print(f"\n  [{i}] scene-{r.scene:04d}  ann_type={r.ann_type}")
    print(f"      Instruction: {r.instruction[:120]}")
    print(f"      GT  end_angle: {gt_a:+6.1f} deg ({cls_dir(gt_a)})")
    print(f"      Top end_angle: {top_a:+6.1f} deg  ADE={r.top_ade:.2f}m")
    print(f"      LLM end_angle: {llm_a:+6.1f} deg  ADE={r.llm_ade:.2f}m  (rank "
          f"{int((np.argsort(-wd['probs']) == int(r.picked_idx)).nonzero()[0][0])+1}/10)")

# ---- 7. Does the LLM agree with itself? -------------------------------------
print("\n" + "=" * 80)
print("7. LLM CONSISTENCY (same window, different annotations -> same pick?)")
print("=" * 80)
multi = df.groupby(['scene', 'pkl']).filter(lambda g: len(g) > 1)
groups = multi.groupby(['scene', 'pkl'])
consist = 0; total_g = 0
for (sc, pkl), g in groups:
    if len(g['picked_idx'].unique()) == 1:
        consist += 1
    total_g += 1
print(f"  Windows with >1 annotation:                {total_g}")
print(f"  Windows where LLM picked the SAME idx:     {consist} ({100*consist/max(1,total_g):.1f}%)")

# Compare to expected if random across 10 with avg 2-3 anns per window
# If picks were random, P(all same | 2 anns) = 1/10, etc.
# So baseline ~10% "consistent" if random, much higher if model is biased.
mean_ann_per_win = df.groupby(['scene','pkl']).size().mean()
print(f"  Mean annotations per window:               {mean_ann_per_win:.2f}")

print("\nDone.")
