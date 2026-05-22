"""Show a sample LLM prompt for a few records from the full run."""
import os, sys, json, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from llm_zero_shot_selector import describe_trajectory

DOC   = os.path.join(HERE, 'documentation')
JSONL = os.path.join(DOC, 'full_val_llm_results.jsonl')
CACHE = os.path.join(DOC, 'inference_cache.pkl')

with open(CACHE, 'rb') as f:
    cache = pickle.load(f)

records = []
with open(JSONL) as f:
    for line in f:
        try:
            records.append(json.loads(line))
        except Exception:
            pass

# Pick 3 representative records: one that improved, one that got worse, one worst-case
def delta(r): return r['top_ade'] - r['llm_ade']
improved = next(r for r in records if delta(r) > 2.0)
worse    = next(r for r in records if delta(r) < -3.0)
# scene-0296 turn-right worst case
worst    = next(r for r in records if r['scene'] == 296)

def show_prompt(r, label):
    wd = cache[(r['scene'], r['pkl'])]
    trajs = wd['trajs']   # (10, 12, 2)
    probs = wd['probs']   # (10,)
    gt    = wd['gt']      # (12, 2)
    dists = wd['dists']   # (10,)

    descs = [describe_trajectory(t) for t in trajs]

    top_k = int(np.argmax(probs))
    llm_k = int(r['picked_idx'])
    best_k = int(np.argmin(dists))

    desc_block = "\n".join(f"  {i+1}. {d}" for i, d in enumerate(descs))

    prompt = (
        "You are choosing the best predicted future trajectory for an autonomous "
        "vehicle. The trajectory must reflect the driver's stated intent.\n\n"
        f"Driver instruction:\n  \"{r['instruction']}\"\n\n"
        f"The model produced 10 candidate 6-second trajectories:\n{desc_block}\n\n"
        "Which trajectory number (1-10) best matches the driver's intent? "
        "Consider direction (left/right/straight), maneuvers (turn, lane change, "
        "stop, slow down), and speed.\n"
        "Reply with ONLY the trajectory number - a single integer between 1 and 10."
    )

    print(f"\n{'='*90}")
    print(f"SAMPLE: {label}")
    print(f"  scene={r['scene']}  ann_type={r['ann_type']}")
    print(f"  top_ADE={r['top_ade']:.2f}m  llm_ADE={r['llm_ade']:.2f}m  "
          f"min_ADE={r['min_ade']:.2f}m  delta={delta(r):+.2f}m")
    print(f"{'='*90}")
    print(prompt)
    print(f"\n--- GROUND TRUTH context (not in prompt) ---")
    print(f"  GT  end: ({gt[-1,0]:+.1f}, {gt[-1,1]:+.1f})  "
          f"end_angle={float(np.degrees(np.arctan2(gt[-1,0],gt[-1,1]))):+.1f} deg")
    print(f"  Model top-k idx: {top_k+1}  prob={probs[top_k]:.3f}  ADE={dists[top_k]:.2f}m")
    print(f"  LLM picked idx:  {llm_k+1}  prob={probs[llm_k]:.3f}  ADE={dists[llm_k]:.2f}m")
    print(f"  Oracle best idx: {best_k+1}  prob={probs[best_k]:.3f}  ADE={dists[best_k]:.2f}m")
    print(f"  Probs: " + "  ".join(f"[{i+1}]={p:.3f}" for i, p in enumerate(probs)))

show_prompt(improved, "IMPROVED (LLM helped)")
show_prompt(worse,    "DEGRADED (LLM hurt)")
show_prompt(worst,    "WORST CASE (scene-0296 turn-right)")
