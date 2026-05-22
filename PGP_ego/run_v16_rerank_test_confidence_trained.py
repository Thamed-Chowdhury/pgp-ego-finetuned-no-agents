"""
V16-style language reranker on the test split, layered over the confidence-trained
PGP-DSVT checkpoint. Same architecture as `pgp-llm-v13/run_v16_rerank.py` (four
symbolic gates + Gemini 2.5 Flash + two post-LLM safety nets), adapted to:

  * use the v1.0-test nuScenes split (not trainval),
  * use the 150 doScenes-style anchors (one per test scene at samples[4]),
  * use the lvm_ranked / stage-2 checkpoint at
    pgp_ego_output_dsvt_ranked_stage2/checkpoints/best.tar,
  * use Linux paths.

The reranker is **text-only** (no images sent to Gemini), exactly as in V16:
  - textual road network derived from the lane graph,
  - the driver instruction,
  - the 10 PGP candidate trajectories with confidence percentages + natural-language
    motion descriptions + lane-graph projection tags.

Stages run in this order, with caches at each step:
  A. inference cache:   pickle of per-anchor (trajs, probs, gt_pgp, per-traj ADE)
  B. test_set:          json of per-anchor (instruction, ann_type, ann_id)
  C. rerank:            jsonl of per-anchor (picked_idx, fallback, reasoning, fix_applied)
  D. submission + self-eval: submission_v16.csv + self_eval_metrics.json

Re-runs reuse caches where present (delete files to force re-execution).
"""

import os
import sys
import csv
import json
import pickle
import time
import re
import warnings
from glob import glob
from math import sqrt

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'pgp-ego-finetuned', 'pgp-llm-v13'))

# Mock get_prediction_challenge_split BEFORE importing the doScenes dataset
# (test split has no prediction-challenge splits file).
import nuscenes.eval.prediction.splits as _ns_pred_splits
_ns_pred_splits.get_prediction_challenge_split = lambda split, dataroot=None: []

from datasets.nuScenes.nuScenes_ego_graphs_doscenes import (
    NuScenesEgoGraphsDoScenes, HISTORY_LEN, FUTURE_LEN, MIN_SCENE_SAMPLES,
)
from train_eval.initialization import initialize_prediction_model
import train_eval.utils as u

from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion

# V16 helpers (reused unchanged)
from lane_graph_utils import (
    extract_lane_graph, add_sibling_connector_edges, add_lane_switch_edges,
    build_reachable_paths_with_waypoints, format_road_network,
    get_ego_pose_from_sample, get_scene_location,
)
from trajectory_projector import project_all, format_trajectory_tags
from trigger_classifier import classify_trigger
from llm_zero_shot_selector import describe_trajectory

import google.genai as genai
from google.genai import types as gtypes


# ────────────────────────────────────────────────────────────────────────────
# Paths / config
# ────────────────────────────────────────────────────────────────────────────
DATA_ROOT      = os.path.join(ROOT, 'nuscenes_data')
TEST_ROOT      = os.path.join(DATA_ROOT, 'v1-test')
TRAINVAL_STATS = os.path.join(ROOT, 'pgp_ego_preprocessed', 'stats.pickle')
TEST_PREPROC   = os.path.join(ROOT, 'pgp_ego_test_preprocessed_dsvt')   # DSVT-injected test pickles
CHECKPOINT     = os.path.join(ROOT, 'pgp_ego_output_dsvt_ranked_stage2', 'checkpoints', 'best.tar')
CONFIG_FILE    = os.path.join(HERE, 'configs', 'pgp_ego_gatx2_lvm_ranked_stage2.yml')
DOSCENES_REPO  = os.path.join(ROOT, 'pgp-ego-finetuned', 'doScenes_repo')
ANN_DIR        = os.path.join(DOSCENES_REPO, 'Annotations')
KEYS_FILE      = os.path.join(HERE, 'Gemini_keys.txt')

OUT_DIR        = os.path.join(ROOT, 'test_v16_rerank_confidence_trained')
INFER_CACHE    = os.path.join(OUT_DIR, 'inference_cache.pkl')
TEST_SET_JSON  = os.path.join(OUT_DIR, 'test_set.json')
RESULTS_JSONL  = os.path.join(OUT_DIR, 'v16_results.jsonl')
SUBMISSION_CSV = os.path.join(OUT_DIR, 'submission.csv')
METRICS_JSON   = os.path.join(OUT_DIR, 'self_eval_metrics.json')

MODEL_ID       = 'gemini-2.5-flash'
CALL_GAP       = 60.0 / 14   # 14 RPM per key
NUM_SAMPLES    = 100         # PGP latent samples at inference

os.makedirs(OUT_DIR, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# Coordinate helpers (reused from run_doscenes_test_baseline.py)
# ────────────────────────────────────────────────────────────────────────────
def pgp_to_challenge_frame(traj_pgp):
    """PGP local (+y forward, +x right)  ->  challenge local (+x forward, +y left)."""
    out = np.empty_like(traj_pgp)
    out[..., 0] = traj_pgp[..., 1]
    out[..., 1] = -traj_pgp[..., 0]
    return out


def scenes_used_for_eval(nusc):
    out = []
    for s in sorted(nusc.scene, key=lambda x: x['name']):
        samples = []
        t = s['first_sample_token']
        while t:
            samples.append(t)
            t = nusc.get('sample', t)['next']
        if len(samples) >= MIN_SCENE_SAMPLES:
            out.append((samples[HISTORY_LEN], s['token'], s['name']))
    return out


def get_anchor_world_pose(nusc, anchor_token):
    sample = nusc.get('sample', anchor_token)
    ld_tok = sample['data']['LIDAR_TOP']
    ld     = nusc.get('sample_data', ld_tok)
    ep     = nusc.get('ego_pose', ld['ego_pose_token'])
    pos = np.array(ep['translation'][:2], dtype=np.float64)
    yaw = float(quaternion_yaw(Quaternion(ep['rotation'])))
    return pos, yaw


def get_future_world_xy(nusc, anchor_token, n_future=FUTURE_LEN):
    cur = nusc.get('sample', anchor_token)
    out = []
    for _ in range(n_future):
        cur = nusc.get('sample', cur['next'])
        ld = nusc.get('sample_data', cur['data']['LIDAR_TOP'])
        ep = nusc.get('ego_pose', ld['ego_pose_token'])
        out.append(ep['translation'][:2])
    return np.array(out, dtype=np.float64)


def world_to_challenge_local(world_xy, anchor_pos, anchor_yaw):
    cos_y, sin_y = np.cos(anchor_yaw), np.sin(anchor_yaw)
    R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    return (R.T @ (world_xy - anchor_pos).T).T


# ────────────────────────────────────────────────────────────────────────────
# Stage A+B — inference + GT cache
# ────────────────────────────────────────────────────────────────────────────
def build_inference_cache(cfg, helper, force=False):
    if os.path.exists(INFER_CACHE) and not force:
        print(f'[infer] reusing {INFER_CACHE}')
        with open(INFER_CACHE, 'rb') as f:
            return pickle.load(f)

    print(f'[infer] running confidence-trained model on {TEST_PREPROC}')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = initialize_prediction_model(
        cfg['encoder_type'], cfg['aggregator_type'], cfg['decoder_type'],
        cfg['encoder_args'], cfg['aggregator_args'], cfg['decoder_args'],
    ).float().to(device)
    model.eval()
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.aggregator.num_samples = NUM_SAMPLES
    model.decoder.num_samples = NUM_SAMPLES
    print(f'  ckpt: {CHECKPOINT}  (val_metric={ckpt.get("val_metric", "?")})')

    sa = dict(cfg['test_set_args'])
    sa['random_flips'] = False
    sa['split'] = 'doscenes_test'

    ds = NuScenesEgoGraphsDoScenes('load_data', TEST_PREPROC, sa, helper)
    anchor_tokens = [t.split('_', 1)[1] for t in ds.token_list]
    print(f'  {len(ds)} test anchors')

    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    cache = {}
    cursor = 0
    nusc = helper.data
    with torch.no_grad():
        for bi, data in enumerate(dl):
            data = u.send_to_device(u.convert_double_to_float(data))
            preds = model(data['inputs'])
            trajs = preds['traj'].detach().cpu().numpy()    # (B, K, T, 2)
            probs = preds['probs'].detach().cpu().numpy()   # (B, K)
            gt    = data['ground_truth']['traj'].detach().cpu().numpy()  # (B, T, 2)
            for b in range(trajs.shape[0]):
                anchor_tok = anchor_tokens[cursor + b]
                per_traj_ade = np.mean(np.linalg.norm(trajs[b] - gt[b][None], axis=-1), axis=-1)
                top_k = int(np.argmax(probs[b]))
                # GT pose for later submission/eval (saves a NuScenes re-lookup)
                anchor_pos, anchor_yaw = get_anchor_world_pose(nusc, anchor_tok)
                future_world = get_future_world_xy(nusc, anchor_tok, FUTURE_LEN)
                future_local = world_to_challenge_local(future_world, anchor_pos, anchor_yaw)
                cache[anchor_tok] = {
                    'trajs':       trajs[b].astype(np.float32),         # (K, T, 2) PGP frame
                    'probs':       probs[b].astype(np.float32),         # (K,) log-softmax for lvm_ranked
                    'gt_pgp':      gt[b].astype(np.float32),            # (T, 2) PGP frame
                    'gt_world':    future_world.astype(np.float64),     # (T, 2) world frame
                    'gt_chal':     future_local.astype(np.float64),     # (T, 2) challenge frame
                    'anchor_pos':  anchor_pos.astype(np.float64),
                    'anchor_yaw':  float(anchor_yaw),
                    'top_k':       top_k,
                    'top_ade':     float(per_traj_ade[top_k]),
                    'min_ade':     float(per_traj_ade.min()),
                    'dists':       per_traj_ade.astype(np.float64),     # (K,) per-traj ADE vs GT
                }
            cursor += trajs.shape[0]
            if (bi + 1) % 5 == 0 or cursor == len(ds):
                print(f'  infer batch {bi+1}/{len(dl)}  ({cursor}/{len(ds)})')

    with open(INFER_CACHE, 'wb') as f:
        pickle.dump(cache, f)
    print(f'[infer] wrote {INFER_CACHE}  ({len(cache)} anchors)')
    return cache


# ────────────────────────────────────────────────────────────────────────────
# Stage C — doScenes annotations
# ────────────────────────────────────────────────────────────────────────────
PRIORITY = {'d': 0, 's': 1, 'sd': 2, 'ds': 3}  # lower = preferred; empty/other are deprioritised

def load_doscenes_annotations():
    """Pool all annotator CSVs, return scene_num -> (instruction, ann_type, ann_id).

    For each scene the best (lowest-priority value) typed annotation across all
    annotators is kept. Empty rows are skipped. Free-form / unknown types are
    kept only if nothing typed exists for that scene.
    """
    by_scene = {}  # scene_num -> (priority, annotator, instruction, ann_type)
    files = sorted(glob(os.path.join(ANN_DIR, '*.csv')))
    print(f'[anno] reading {len(files)} annotator csvs from {ANN_DIR}')
    for fp in files:
        annotator = os.path.basename(fp).replace('doScenesAnnotations - ', '').replace('.csv', '')
        with open(fp, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    scene_num = int(str(row.get('Scene Number', '')).strip())
                except Exception:
                    continue
                instr = (row.get('Instruction') or '').strip()
                if not instr:
                    continue
                ann_type = (row.get('Instruction Type') or '').strip().lower().rstrip(' "')
                pr = PRIORITY.get(ann_type, 9)  # 9 = unknown/free-form
                cur = by_scene.get(scene_num)
                if (cur is None) or (pr < cur[0]):
                    by_scene[scene_num] = (pr, annotator, instr, ann_type or '?')
    print(f'  pooled annotations for {len(by_scene)} unique scene numbers')
    return by_scene


def assemble_test_set(scene_tuples, cache, annotations, force=False):
    """For each test anchor, look up the best doScenes annotation by scene number
    (the trailing integer of 'scene-XXXX'). Anchors with no annotation get an
    empty instruction and will hit the auto-fallback (top-conf) at rerank time.
    """
    if os.path.exists(TEST_SET_JSON) and not force:
        print(f'[test_set] reusing {TEST_SET_JSON}')
        with open(TEST_SET_JSON) as f:
            return json.load(f)

    records = []
    no_ann = 0
    for anchor_tok, scene_tok, scene_name in scene_tuples:
        try:
            scene_num = int(scene_name.split('-')[1])
        except Exception:
            scene_num = -1
        wd = cache.get(anchor_tok)
        if wd is None:
            continue
        ann = annotations.get(scene_num)
        if ann is None:
            no_ann += 1
            instr = ''
            ann_type = ''
            annotator = ''
        else:
            _, annotator, instr, ann_type = ann
        records.append({
            'scene':       scene_name,
            'scene_num':   scene_num,
            'scene_token': scene_tok,
            'anchor_token': anchor_tok,
            'ann_id':      f'{annotator}:{scene_num}',
            'ann_type':    ann_type,
            'instruction': instr,
            'picked_idx':  wd['top_k'],
            'fallback':    False,
            'llm_ade':     wd['top_ade'],
            'top_ade':     wd['top_ade'],
            'min_ade':     wd['min_ade'],
        })
    with open(TEST_SET_JSON, 'w') as f:
        json.dump(records, f, indent=2)
    print(f'[test_set] wrote {TEST_SET_JSON}  ({len(records)} anchors, {no_ann} with no annotation)')
    return records


# ────────────────────────────────────────────────────────────────────────────
# Stage D — V16 reranking
# ────────────────────────────────────────────────────────────────────────────
HAS_EXPLICIT_DIR = re.compile(
    r'\b(turn|take|go|make|bear|head|veer)\s+(a\s+|an\s+)?'
    r'(sharp\s+|hard\s+|slight\s+)?(left|right)\b'
    r'|\b(lane\s+change|switch\s+lanes?|change\s+lanes?|merge'
    r'|slide\s+into\s+(the\s+)?(left|right)\s+lane)\b',
    re.IGNORECASE,
)
FUTURE_INTENT_RE = re.compile(
    r'\b(and\s+then|then\s+(take|turn|make|go|continue)|after\s+(that|the\s+\w+)'
    r'|once\s+(you|the|clear|traffic)|before\s+(turning|the\s+\w+))\b',
    re.IGNORECASE,
)
CURRENT_KEEP_RE = re.compile(
    r'^\s*(keep|continue|stay|maintain|proceed|go\s+straight|'
    r'go\s+forward|drive\s+straight)\b',
    re.IGNORECASE,
)
POST_TURN_RE = re.compile(
    r'\b(then\s+continue|after\s+the\s+(turn|intersection)'
    r'|once\s+(through|past)|complete\s+the\s+turn)\b',
    re.IGNORECASE,
)


SYS_MSG = (
    "You are an expert autonomous driving trajectory selector.\n"
    "You will see (a) a textual road network around the ego vehicle, "
    "(b) a driver instruction, and (c) 10 candidate predicted trajectories "
    "with their lane-graph projections.\n\n"
    "IMPORTANT: The trajectories are listed from HIGHEST to LOWEST model "
    "confidence (Traj 1 = PGP's top-confidence prediction, Traj 10 = lowest "
    "confidence). Each trajectory shows its confidence percentage.\n\n"
    "Your job: pick the SINGLE candidate that best matches the driver's "
    "intent AND stays on a reachable lane path.\n\n"
    "Rules:\n"
    "1. DEFAULT TO TRAJ 1. It is the model's best prediction. Only choose "
    "a different trajectory if the driver instruction explicitly specifies "
    "a direction or maneuver that Traj 1 does NOT match (e.g., the "
    "instruction says 'turn left' but Traj 1 goes straight or right).\n"
    "2. Direction override: If the instruction clearly specifies LEFT, RIGHT, "
    "or a lane change, prefer the candidate whose lane projection matches "
    "that direction, even if its confidence is lower than Traj 1.\n"
    "3. Prefer on-road candidates over off-road ones.\n"
    "4. If no candidate matches the requested direction, fall back to Traj 1.\n"
    "5. If the instruction is conditional ('when X', 'after X') and the "
    "ego is currently stopped or crawling, the trigger may NOT yet be met. "
    "In that case, prefer the candidate that maintains current motion.\n\n"
    "Output FORMAT (exact):\n"
    "REASONING: <your analysis - state whether Traj 1 matches the instruction "
    "and why you are or are not deviating from it>\n"
    "PICK: <single integer 1-10>\n"
)


def find_working_keys(keys_file, model_id):
    with open(keys_file) as f:
        all_keys = [k.strip() for k in f if k.strip()]
    print(f'[gemini] pre-flight testing {len(all_keys)} keys against {model_id} ...')
    working = []
    for i, k in enumerate(all_keys):
        try:
            c = genai.Client(api_key=k)
            c.models.generate_content(
                model=model_id, contents='Reply with OK',
                config=gtypes.GenerateContentConfig(temperature=0, max_output_tokens=8),
            )
            working.append(k)
            print(f'  key {i:2d}: OK')
        except Exception as e:
            print(f'  key {i:2d}: FAIL  {str(e)[:80]}')
        time.sleep(0.3)
    return working


def query_llm(prompt, sys_msg, clients, key_last_call, model_id, gap,
              max_output_tokens=2048, retries=2):
    for attempt in range(retries * len(clients)):
        now = time.time()
        idx = max(range(len(clients)), key=lambda i: now - key_last_call[i])
        elapsed = now - key_last_call[idx]
        if elapsed < gap:
            time.sleep(gap - elapsed)
        key_last_call[idx] = time.time()
        try:
            resp = clients[idx].models.generate_content(
                model=model_id, contents=prompt,
                config=gtypes.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=max_output_tokens,
                    system_instruction=sys_msg,
                ),
            )
            return resp.text or ''
        except Exception as e:
            err = str(e)
            wait = 5.0 if ('quota' in err.lower() or '429' in err) else 2.0
            time.sleep(wait)
    return ''


def parse_pick(response):
    if not response:
        return None
    m = re.search(r'PICK\s*:\s*(\d+)', response, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 10:
            return n - 1
    m = re.search(r'\b([1-9]|10)\b', response)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 10:
            return n - 1
    return None


def make_user_prompt(instruction, road_text, traj_descs, traj_tags,
                     last_speed, trigger_status, conf_pcts):
    desc_block = "\n".join(
        f"  Traj {i+1} (conf {conf_pcts[i]}): {d}"
        for i, d in enumerate(traj_descs)
    )
    return (
        f"DRIVER INSTRUCTION: \"{instruction}\"\n"
        f"  Trigger status (heuristic): {trigger_status}\n"
        f"  Current ego speed: {last_speed:.1f} m/s\n\n"
        f"{road_text}\n\n"
        f"=== CANDIDATE TRAJECTORIES (sorted highest->lowest confidence) ===\n"
        f"{desc_block}\n\n"
        f"=== CANDIDATE TRAJECTORIES (lane-graph projection) ===\n"
        f"{traj_tags}\n\n"
        f"Remember: default to Traj 1 unless the instruction requires a "
        f"different direction. Now choose the best candidate."
    )


def should_force_top_conf(instruction, ann_type, ego_on_connector):
    """V16 pre-LLM gates. Returns (force, reason)."""
    if not instruction:
        return True, 'V16-0: no annotation for this scene'
    if ann_type == '?':
        return True, "FixV16-1: ann_type='?'"
    if ego_on_connector and POST_TURN_RE.search(instruction):
        return True, "FixV16-4: connector+post-turn"
    if FUTURE_INTENT_RE.search(instruction):
        parts = re.split(
            r'\b(and\s+then|then|after|once|before)\b',
            instruction, maxsplit=1, flags=re.IGNORECASE,
        )
        first = parts[0].strip() if parts else instruction
        if CURRENT_KEEP_RE.search(first):
            return True, "FixV16-3: future-intent w/ keep-going prefix"
    return False, ''


def get_last_speed(test_preproc, anchor_tok):
    """Read the last-history-step longitudinal speed from the per-window pickle."""
    pkl_path = os.path.join(test_preproc, f'ego_{anchor_tok}.pickle')
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    ta = raw['inputs']['target_agent_representation']
    return float(ta[-1, 2])


def run_v16_rerank(test_records, cache, nusc, map_cache, clients, key_last_call, force=False):
    if os.path.exists(RESULTS_JSONL) and not force:
        print(f'[rerank] reusing {RESULTS_JSONL}')
        with open(RESULTS_JSONL) as f:
            return [json.loads(l) for l in f]

    results = []
    n_active = n_fallback = n_failed = 0
    t0 = time.time()
    out = open(RESULTS_JSONL, 'w', encoding='utf-8')

    for i, rec in enumerate(test_records):
        anchor_tok = rec['anchor_token']
        wd = cache[anchor_tok]
        sc_name = rec['scene']

        try:
            last_speed = get_last_speed(TEST_PREPROC, anchor_tok)
        except Exception as e:
            print(f'  [{i+1}/{len(test_records)}] {sc_name} ERROR speed read: {e}')
            n_failed += 1; continue

        sample = nusc.get('sample', anchor_tok)
        location = get_scene_location(nusc, sample['scene_token'])
        if location not in map_cache:
            print(f'    loading NuScenesMap for {location}...')
            map_cache[location] = NuScenesMap(dataroot=TEST_ROOT, map_name=location)
        nusc_map = map_cache[location]
        ex, ey, heading = get_ego_pose_from_sample(nusc, anchor_tok)

        graph = extract_lane_graph(nusc_map, ex, ey, heading, radius=100.0)
        ego_lane_id = ''
        ego_on_connector = False
        paths_wp = []
        road_text = '(no lanes in radius)'
        projections_sorted = []
        traj_tags = ''
        if graph is not None:
            add_sibling_connector_edges(graph)
            add_lane_switch_edges(graph, lateral_thresh=6.0, heading_thresh_deg=30.0)
            paths_wp = build_reachable_paths_with_waypoints(graph, wp_spacing=5.0)
            road_text = format_road_network(graph, paths_wp)
            ego_lane_id = graph.get('ego_lane') or ''
            ego_on_connector = ego_lane_id.startswith('C')
            projections = project_all(wd['trajs'], graph, paths_wp)
        else:
            projections = [
                {'lane_seq': [], 'matched_path_idx': None, 'matched_direction': None,
                 'mean_offroad': 0.0, 'max_offroad': 0.0, 'n_offroad': 0,
                 'visit_counts': {}}
                for _ in range(len(wd['trajs']))
            ]

        trigger = classify_trigger(rec['instruction'], last_speed, ego_on_connector)

        # Sort PGP candidates by confidence (highest -> lowest)
        probs = np.array(wd['probs'])
        rank_order = np.argsort(-probs)
        conf_pcts = []
        for idx in rank_order:
            p = float(probs[idx])
            # If probs are log-softmax (LVMRanked), convert; else assume already in [0,1].
            if p <= 0.0:
                p = float(np.exp(p))
            conf_pcts.append(f'{100*p:.0f}%')
        projections_sorted = [projections[i] for i in rank_order]
        traj_tags = format_trajectory_tags(projections_sorted) if graph is not None else '(no projection)'
        traj_descs = [describe_trajectory(wd['trajs'][i]) for i in rank_order]

        fix_applied = []
        force_top, reason = should_force_top_conf(rec['instruction'], rec['ann_type'], ego_on_connector)

        if force_top or trigger == 'PENDING' or graph is None:
            picked = int(wd['top_k'])
            fb_reason = reason if force_top else ('PENDING (auto-skip LLM)' if trigger == 'PENDING' else 'no lane graph')
            if force_top:
                fix_applied.append(reason)
            response_text = ''
            new_ade = float(wd['top_ade'])
            n_fallback += 1
            llm_called = False
        else:
            user_prompt = make_user_prompt(
                rec['instruction'], road_text, traj_descs, traj_tags,
                last_speed, trigger, conf_pcts,
            )
            response_text = query_llm(user_prompt, SYS_MSG, clients, key_last_call,
                                      MODEL_ID, gap=CALL_GAP, max_output_tokens=2048)
            picked_sorted = parse_pick(response_text)
            llm_called = True

            if picked_sorted is None:
                picked = int(wd['top_k'])
                fb_reason = 'LLM parse failed'
                n_fallback += 1
            else:
                # Post-LLM safety: snap-back if no explicit direction trigger
                if not HAS_EXPLICIT_DIR.search(rec['instruction']):
                    if picked_sorted != 0:
                        fix_applied.append(f'FixV16-2: {picked_sorted+1}->1 (no explicit dir)')
                    picked_sorted = 0

                # Fix B: upgrade to highest-confidence same-direction candidate
                if picked_sorted > 0:
                    picked_dir = projections_sorted[picked_sorted].get('matched_direction')
                    if picked_dir is not None:
                        for higher_rank in range(picked_sorted):
                            if projections_sorted[higher_rank].get('matched_direction') == picked_dir:
                                fix_applied.append(
                                    f'FixB: {picked_sorted+1}->{higher_rank+1} (dir={picked_dir})')
                                picked_sorted = higher_rank
                                break

                picked = int(rank_order[picked_sorted])
                fb_reason = ''
                n_active += 1
            new_ade = float(wd['dists'][picked])

        result = {
            'scene':            sc_name,
            'anchor_token':     anchor_tok,
            'ann_type':         rec['ann_type'],
            'instruction':      rec['instruction'],
            'trigger_status':   trigger,
            'last_speed':       last_speed,
            'ego_on_connector': ego_on_connector,
            'picked_idx':       picked,
            'top_idx':          int(wd['top_k']),
            'new_ade':          new_ade,
            'top_ade':          float(wd['top_ade']),
            'min_ade':          float(wd['min_ade']),
            'llm_called':       llm_called,
            'fallback_reason':  fb_reason,
            'fix_applied':      fix_applied,
            'reasoning':        (response_text or '')[:400],
        }
        results.append(result)
        out.write(json.dumps(result) + '\n'); out.flush()

        if (i + 1) % 10 == 0 or (i + 1) == len(test_records):
            elapsed = time.time() - t0
            print(f'  [{i+1:3d}/{len(test_records)}] active={n_active} fallback={n_fallback} '
                  f'failed={n_failed}  elapsed={elapsed:.0f}s')

    out.close()
    print(f'[rerank] DONE: active={n_active} fallback={n_fallback} failed={n_failed}')
    return results


# ────────────────────────────────────────────────────────────────────────────
# Stage E — submission CSV + self-eval
# ────────────────────────────────────────────────────────────────────────────
def write_submission_and_eval(results, cache, scene_tuples, helper, map_cache):
    header = ['sample_token'] + [f'x{i}' for j in range(1, FUTURE_LEN + 1) for i in [f'{j}']] # placeholder
    header = ['sample_token']
    for i in range(1, FUTURE_LEN + 1):
        header += [f'x{i}', f'y{i}']

    by_anchor = {r['anchor_token']: r for r in results}
    rows = []
    for anchor_tok, scene_tok, scene_name in scene_tuples:
        r = by_anchor.get(anchor_tok)
        wd = cache.get(anchor_tok)
        if r is None or wd is None:
            continue
        pick_idx = r['picked_idx']
        chal = pgp_to_challenge_frame(wd['trajs'][pick_idx])
        row = [scene_tok]
        for x, y in chal:
            row.extend([f'{float(x):.6f}', f'{float(y):.6f}'])
        rows.append(row)

    with open(SUBMISSION_CSV, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f'[submit] wrote {SUBMISSION_CSV}  ({len(rows)} rows)')

    # Self-eval with doScenes_repo/metrics.py
    import importlib.util
    metrics_path = os.path.join(DOSCENES_REPO, 'metrics.py')
    spec = importlib.util.spec_from_file_location('doscenes_metrics', metrics_path)
    doscenes_metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(doscenes_metrics)
    compute_ego_metrics = doscenes_metrics.compute_ego_metrics

    nusc = helper.data
    per_scene = []
    for anchor_tok, scene_tok, scene_name in scene_tuples:
        r = by_anchor.get(anchor_tok)
        wd = cache.get(anchor_tok)
        if r is None or wd is None:
            continue
        location = get_scene_location(nusc, scene_tok)
        if location not in map_cache:
            print(f'    [self-eval] loading map {location}')
            map_cache[location] = NuScenesMap(dataroot=TEST_ROOT, map_name=location)
        nusc_map = map_cache[location]
        pick_idx = r['picked_idx']
        pred_local = pgp_to_challenge_frame(wd['trajs'][pick_idx])
        m = compute_ego_metrics(pred_local, wd['gt_chal'], wd['anchor_pos'], wd['anchor_yaw'], nusc_map)
        per_scene.append({'scene': scene_name, 'anchor_token': anchor_tok,
                          'picked_idx': pick_idx, 'top_idx': int(wd['top_k']),
                          'instruction': r['instruction'], 'metrics': m})

    keys = ['ade_2s', 'ade_4s', 'ade_6s', 'fde', 'miss_rate',
            'speed_error', 'ahe', 'fhe', 'offroad', 'offroad_rate', 'offyaw']
    agg = {k: float(np.mean([s['metrics'][k] for s in per_scene])) for k in keys}
    with open(METRICS_JSON, 'w') as f:
        json.dump({'aggregate': agg, 'per_scene': per_scene, 'n_scenes': len(per_scene)}, f, indent=2)
    print(f'[self-eval] wrote {METRICS_JSON}')
    return agg, per_scene


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    print('='*78); print('V16 reranker on test split, confidence-trained PGP-DSVT'); print('='*78)
    with open(CONFIG_FILE) as f:
        cfg = yaml.safe_load(f)
    cfg['version'] = 'v1.0-test'

    nusc = NuScenes(version='v1.0-test', dataroot=TEST_ROOT, verbose=False)
    helper = PredictHelper(nusc)
    print(f'[main] {len(nusc.scene)} test scenes loaded')

    scene_tuples = scenes_used_for_eval(nusc)
    print(f'[main] {len(scene_tuples)} scenes meet >= {MIN_SCENE_SAMPLES} samples')

    cache = build_inference_cache(cfg, helper)
    annotations = load_doscenes_annotations()
    test_records = assemble_test_set(scene_tuples, cache, annotations)

    print(f'\n[main] pre-flight Gemini keys')
    keys = find_working_keys(KEYS_FILE, MODEL_ID)
    if not keys:
        print('ERROR: no working keys')
        sys.exit(2)
    clients = [genai.Client(api_key=k) for k in keys]
    key_last_call = [0.0] * len(clients)
    print(f'[main] {len(keys)} working keys')

    map_cache = {}
    results = run_v16_rerank(test_records, cache, nusc, map_cache, clients, key_last_call)

    agg, per_scene = write_submission_and_eval(results, cache, scene_tuples, helper, map_cache)

    print('\n' + '='*78)
    print('FINAL — V16 reranker on top of confidence-trained PGP-DSVT')
    print('='*78)
    print(f'  N scenes evaluated: {len(per_scene)}')
    for k in ['ade_2s', 'ade_4s', 'ade_6s', 'fde', 'miss_rate',
              'speed_error', 'ahe', 'fhe', 'offroad', 'offroad_rate', 'offyaw']:
        print(f'  {k:14s} = {agg[k]:.4f}')
    print('='*78)

    # Per-scene rerank statistics
    n_llm    = sum(1 for r in results if r['llm_called'])
    n_change = sum(1 for r in results if r['picked_idx'] != r['top_idx'])
    n_anno   = sum(1 for r in results if r['instruction'])
    print(f'  scenes with doScenes annotation:   {n_anno}')
    print(f'  scenes sent to Gemini 2.5 Flash:   {n_llm}')
    print(f'  scenes where pick changed vs top:  {n_change}')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
