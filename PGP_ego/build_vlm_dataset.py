"""
VLM Finetuning Dataset Builder — gemma-4-31b-it reasoning for trajectory ranking.

For each annotated val scene:
  1. Run PGP inference → K=10 trajectories + confidence scores.
  2. Rank trajectories by mean ADE vs GT (rank 1 = closest to GT).
  3. Extract road-network text via lane_graph_utils.py.
  4. Load CAM_FRONT images for the last 2 seconds (4 keyframes).
  5. Build multimodal prompt → ask gemma-4-31b-it for detailed ranking reasoning.
  6. Save each (scene, instruction) pair as one JSONL record.

Usage:
  cd /teamspace/studios/this_studio
  python3 -u PGP_ego/build_vlm_dataset.py [--high_ade_only] [--out_dir vlm_dataset]
"""

import argparse, os, sys, json, pickle, re, threading, time, random, math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

os.environ.setdefault('RAY_memory_monitor_refresh_ms', '0')
sys.path.insert(0, 'PGP_ego')
sys.path.insert(0, 'pgp-ego-finetuned/pgp-llm-v13')

import google.genai as genai
from google.genai import types as gtypes

from train_eval.initialization import initialize_prediction_model
import train_eval.utils as u
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

from lane_graph_utils import (
    extract_lane_graph, add_sibling_connector_edges, add_lane_switch_edges,
    build_reachable_paths_with_waypoints, format_road_network,
    get_ego_pose_from_sample, get_scene_location,
)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT   = 'nuscenes_data'
PREPROC_DIR = 'pgp_ego_preprocessed'
ANNOT_CSV   = 'annotated_doscenes.csv'
KEYS_FILE   = 'PGP_ego/Gemini_keys.txt'
CFG_FILE    = 'PGP_ego/configs/pgp_ego_gatx2_lvm_traversal.yml'
CHECKPOINT  = 'pgp_ego_output2/checkpoints/best.tar'
MAPS_DIR    = os.path.join(DATA_ROOT, 'maps')

HIGH_ADE = [44, 297, 298, 285, 165, 56, 67, 45, 68, 211, 220,
            292, 58, 42, 284, 27, 172, 124, 154]

CALL_GAP  = 60.0 / 14   # 14 RPM per key
CALL_TIMEOUT = 120.0    # seconds — abandon a Gemma call that hangs past this
NUM_SAMPLES = 100        # LVM samples for inference

T_PER_STEP = 0.5         # seconds per trajectory step
T_F        = 6.0         # 6-second future horizon
N_STEPS    = int(T_F / T_PER_STEP)   # 12 steps

NUSC_MAPS  = {}          # lazy cache: location -> NuScenesMap

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── Reasoning prompt template ─────────────────────────────────────────────────
# Teacher-aware prompt: Gemma SEES the GT-derived ranking and per-trajectory
# ADE so it can produce accurate reasoning (Gemma alone can't reliably rerank
# 10 trajectories from instruction + scene). The prompt then INSTRUCTS Gemma
# to write its reasoning AS IF it deduced the ranking itself — never quoting
# the ADE values, never naming "ground truth", never acknowledging that the
# answer was given. This way the student VLM trains on text whose VOICE
# matches the inference-time setting (no GT) while the CONTENT remains
# anchored to the correct ranking.
TEACHER_REASONING_PROMPT = """\
You are an expert autonomous driving system analyst.

You will be given:
- A driver's natural-language instruction.
- The road network ahead, described as text.
- Recent forward (CAM_FRONT) camera images, when available.
- K=10 candidate future trajectories produced by a neural network.
- The CORRECT ranking of those candidates, derived from a future ground-truth \
trajectory NOT shown to you.

Your job is to write a detailed natural-language EXPLANATION of why this \
ranking is correct, grounded ONLY in the driver instruction, the road \
network, the camera images, and the geometric/behavioral features of each \
candidate trajectory. Your explanation will be used as training data for a \
student vision-language model that will receive the SAME inputs as you, \
MINUS the ranking, and must produce the same ranking + reasoning itself.

CRITICAL RULES for your explanation (the student will not have these inputs):
- Do NOT use the words "ADE", "GT", "ground truth", "ground-truth", or any \
synonym that reveals a future trajectory was given to you.
- Do NOT say things like "I was told", "I was given", "the correct answer", \
"the ranking provided", "the ranking I was shown", or any phrase \
acknowledging that the ranking was supplied.
- Do NOT cite numeric distance-to-future-trajectory values.
- Write as if YOU derived the ranking from the instruction, the road \
network, the camera images, and the candidate-trajectory geometry alone.
- Reference candidates by their #index (e.g. "#3") and describe their \
direction, curvature, speed profile, and endpoint.

=== INPUTS ===

DRIVER INSTRUCTION:
"{instruction}"

{road_network_block}

CANDIDATE TRAJECTORIES (K=10, from neural network):
All trajectories are in ego-relative coordinates.
  lateral:      + right, − left  (metres)
  longitudinal: + ahead, − behind (metres)
  Duration: 6 seconds (12 time steps × 0.5 s)

{traj_block}

CORRECT RANKING (best → worst) — known only to you, do not reveal:
{ranking_block}

=== YOUR OUTPUT ===

Provide ~200-400 words of analytical reasoning that:
1. Interprets what the driver instruction implies about the intended maneuver.
2. Reads the road network and any camera images to identify available paths \
and constraints.
3. Explains why the top-ranked candidates fit the instruction and scene.
4. Explains why the lower-ranked candidates do not fit.
5. Names the geometric or behavioral signals that distinguish good \
candidates from bad ones.

Start directly with the analysis. Do not begin with a meta-comment about \
the task. Do not write a header line such as "REASONING:" — just the prose.
"""

# Same template, but the CORRECT RANKING block and the "known only to you"
# clause are removed. This is what the student will see at SFT/inference time.
STUDENT_REASONING_PROMPT = """\
You are an expert autonomous driving system analyst.

You will be given:
- A driver's natural-language instruction.
- The road network ahead, described as text.
- Recent forward (CAM_FRONT) camera images, when available.
- K=10 candidate future trajectories produced by a neural network.

Your job is to rank the candidates from best to worst based on how well \
each matches the driver instruction and the road structure visible in the \
camera images, and to write a detailed natural-language explanation of your \
ranking grounded in the instruction, road network, camera images, and the \
geometric/behavioral features of each candidate trajectory.

=== INPUTS ===

DRIVER INSTRUCTION:
"{instruction}"

{road_network_block}

CANDIDATE TRAJECTORIES (K=10, from neural network):
All trajectories are in ego-relative coordinates.
  lateral:      + right, − left  (metres)
  longitudinal: + ahead, − behind (metres)
  Duration: 6 seconds (12 time steps × 0.5 s)

{traj_block}

=== YOUR OUTPUT ===

RANKING: <comma-separated trajectory indices, best first, e.g. 3,7,1,5,2,9,4,8,6,10>

Then provide ~200-400 words of analytical reasoning that:
1. Interprets what the driver instruction implies about the intended maneuver.
2. Reads the road network and any camera images to identify available paths \
and constraints.
3. Explains why the top-ranked candidates fit the instruction and scene.
4. Explains why the lower-ranked candidates do not fit.
5. Names the geometric or behavioral signals that distinguish good \
candidates from bad ones.

Reference candidates by their #index (e.g. "#3") and describe their \
direction, curvature, speed profile, and endpoint.
"""


# Phrases that, if they appear in Gemma's output, mean the teacher leaked
# knowledge of the GT trajectory / given ranking. Used as a QC signal — not
# enforced at write time; downstream filtering can drop or keep these.
_GT_LEAK_PATTERNS = [
    r'\bADEs?\b',
    r'\bGT\b',
    r'ground[\s\-]?truth',
    r'\bi (?:was|am) (?:told|given|shown|provided|informed)\b',
    r'\b(?:given|provided|supplied) (?:ranking|answer|order)\b',
    r'\bthe (?:correct|right|true) (?:ranking|answer|order)\b',
    r'\bthe ranking (?:i|you|we) (?:was |were |am )?(?:given|told|shown|provided)\b',
    r'\baccording to the (?:ranking|answer)\b',
    r'\bas (?:noted|stated|indicated) in the ranking\b',
    r'\bfuture (?:ground[\s\-]?truth )?trajectory\b',
]
_GT_LEAK_RE = re.compile('|'.join(_GT_LEAK_PATTERNS), re.IGNORECASE)


def detect_gt_leak(text: str) -> list:
    """Return the list of forbidden phrases found in `text` (lowercased, deduped)."""
    if not text:
        return []
    hits = _GT_LEAK_RE.findall(text)
    return sorted({h.lower().strip() for h in hits if h})


# ── Helpers ───────────────────────────────────────────────────────────────────
def describe_trajectory(traj: np.ndarray) -> str:
    """Return a one-line text description of a (T, 2) ego-frame trajectory."""
    diffs = np.diff(traj, axis=0)
    step_dists = np.linalg.norm(diffs, axis=-1)
    speeds = step_dists / T_PER_STEP

    init_v  = float(speeds[:3].mean())
    final_v = float(speeds[-3:].mean())
    total   = float(step_dists.sum())
    x_f, y_f = float(traj[-1, 0]), float(traj[-1, 1])
    end_angle = float(np.degrees(np.arctan2(x_f, y_f)))
    max_lat = float(np.max(np.abs(traj[:, 0])))

    if total < 3.0:
        direction = "barely moves (likely stops)"
    elif abs(end_angle) < 4:
        direction = "goes straight"
    elif abs(end_angle) < 10:
        direction = f"mostly straight with slight {'left' if end_angle < 0 else 'right'}ward drift"
    elif abs(end_angle) < 22:
        direction = f"curves {'left' if end_angle < 0 else 'right'} ~{abs(end_angle):.0f}°"
    else:
        direction = f"turns {'left' if end_angle < 0 else 'right'} {abs(end_angle):.0f}°"

    if abs(end_angle) < 10 and max_lat > 1.5:
        side = 'left' if traj[np.argmax(np.abs(traj[:, 0])), 0] < 0 else 'right'
        direction += f" with {max_lat:.1f}m {side}ward shift"

    if max(init_v, final_v) < 0.8:
        speed = "stationary"
    elif abs(final_v - init_v) < 1.0:
        speed = f"~constant {init_v:.1f} m/s"
    elif final_v < init_v - 1.0:
        speed = f"decelerates {init_v:.1f}→{final_v:.1f} m/s"
    else:
        speed = f"accelerates {init_v:.1f}→{final_v:.1f} m/s"

    endpoint = f"endpoint ({x_f:+.1f}, {y_f:+.1f})m"
    return f"{direction}; {total:.0f}m total; {speed}; {endpoint}"


def ade_k(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Compute per-trajectory ADE. pred:(K,T,2), gt:(T,2) -> (K,)"""
    return np.linalg.norm(pred - gt[None], axis=-1).mean(axis=1)


def load_model(cfg: dict) -> torch.nn.Module:
    model = initialize_prediction_model(
        cfg['encoder_type'], cfg['aggregator_type'], cfg['decoder_type'],
        cfg['encoder_args'], cfg['aggregator_args'], cfg['decoder_args'],
    ).float().to(device)
    model.eval()
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.aggregator.num_samples = NUM_SAMPLES
    model.decoder.num_samples = NUM_SAMPLES
    return model


def run_inference(model: torch.nn.Module, pkl_path: str):
    """Returns (trajs, probs, gt, sample_token) from one preprocessed pickle."""
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    data_t = u.send_to_device(u.convert_double_to_float(u.convert2tensors(raw)))
    sample_token = raw['inputs']['sample_token']
    with torch.no_grad():
        preds = model(data_t['inputs'])
    trajs = preds['traj'][0].detach().cpu().numpy()    # (K, T, 2)
    probs = preds['probs'][0].detach().cpu().numpy()   # (K,)
    gt    = data_t['ground_truth']['traj'][0].detach().cpu().numpy()  # (T, 2)
    return trajs, probs, gt, sample_token


def build_scene_pickle_map(preproc_dir: str) -> dict:
    """Map scene_num -> list of pickle paths for ANY annotated scene with pickles."""
    tv_dir = os.path.join(DATA_ROOT, 'v1.0-trainval')
    with open(os.path.join(tv_dir, 'sample.json')) as f:
        samples = json.load(f)
    with open(os.path.join(tv_dir, 'scene.json')) as f:
        scenes_j = json.load(f)
    sc_tok_to_name = {s['token']: s['name'] for s in scenes_j}
    sample_to_sc   = {s['token']: sc_tok_to_name.get(s['scene_token'], '') for s in samples}
    out = {}
    for fname in sorted(os.listdir(preproc_dir)):
        if not fname.endswith('.pickle') or fname == 'stats.pickle':
            continue
        tok = fname[4:].replace('.pickle', '')
        sc_name = sample_to_sc.get(tok, '')
        if not sc_name:
            continue
        sc_num = int(sc_name.split('-')[1])
        out.setdefault(sc_num, []).append(os.path.join(preproc_dir, fname))
    return out


def get_cam_front_images(nusc: NuScenes, sample_token: str, n_frames: int = 4):
    """Return list of (path, exists) for the last n_frames CAM_FRONT keyframes."""
    results = []
    tok = sample_token
    for _ in range(n_frames):
        if not tok:
            break
        sample = nusc.get('sample', tok)
        sd = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        fpath = os.path.join(DATA_ROOT, sd['filename'])
        results.append(fpath)
        tok = sample['prev']
    results.reverse()   # chronological order (oldest first)
    return results


def load_image_bytes(path: str) -> bytes | None:
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return f.read()
    return None


def get_nusc_map(nusc: NuScenes, scene_token: str) -> NuScenesMap:
    location = get_scene_location(nusc, scene_token)
    if location not in NUSC_MAPS:
        NUSC_MAPS[location] = NuScenesMap(dataroot=DATA_ROOT, map_name=location)
    return NUSC_MAPS[location]


def build_road_network_text(nusc: NuScenes, sample_token: str, scene_token: str) -> str:
    try:
        ego_x, ego_y, heading = get_ego_pose_from_sample(nusc, sample_token)
        nusc_map = get_nusc_map(nusc, scene_token)
        graph = extract_lane_graph(nusc_map, ego_x, ego_y, heading, radius=80.0)
        if graph is None:
            return "(road network unavailable: off-map)"
        add_sibling_connector_edges(graph)
        add_lane_switch_edges(graph)
        paths_wp = build_reachable_paths_with_waypoints(graph, wp_spacing=5.0)
        return format_road_network(graph, paths_wp)
    except Exception as e:
        return f"(road network error: {e})"


def find_working_keys(keys_file: str) -> list:
    with open(keys_file) as f:
        all_keys = [k.strip() for k in f if k.strip()]
    print(f"  Pre-flight testing {len(all_keys)} Gemini keys ...", flush=True)
    working = []
    for i, k in enumerate(all_keys):
        c = genai.Client(api_key=k)
        for _ in range(2):
            try:
                c.models.generate_content(
                    model='gemma-4-31b-it', contents='test',
                    config=gtypes.GenerateContentConfig(temperature=0, max_output_tokens=4),
                )
                working.append(k)
                print(f"    Key {i:2d}: OK", flush=True)
                break
            except Exception as e:
                err = str(e)
                if any(x in err for x in ('403', '400', '429')) or 'quota' in err.lower():
                    print(f"    Key {i:2d}: skip ({err[:40]})", flush=True)
                    break
                time.sleep(2)
        time.sleep(0.3)
    return working


def _generate_with_timeout(client, contents, config, timeout: float):
    """Run client.models.generate_content in a daemon thread with a hard
    wall-clock timeout. Raises TimeoutError if the call doesn't return in time
    — the underlying HTTP request is abandoned (the daemon thread will exit
    when it eventually unblocks or when the process exits)."""
    box = {}
    def runner():
        try:
            box['resp'] = client.models.generate_content(
                model='gemma-4-31b-it', contents=contents, config=config,
            )
        except BaseException as e:
            box['err'] = e
    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError(f"Gemma call exceeded {timeout:.0f}s")
    if 'err' in box:
        raise box['err']
    return box.get('resp')


def call_gemma(clients, key_last_call, contents, max_tokens=1024) -> str:
    """Call gemma-4-31b-it with key rotation. Returns text or '' on failure.

    Each API call has a hard CALL_TIMEOUT wall-clock cap so a stuck request
    cannot freeze the whole run — on timeout we move to the next attempt
    (different key after rotation)."""
    config = gtypes.GenerateContentConfig(
        temperature=0.3, max_output_tokens=max_tokens,
    )
    for attempt in range(len(clients) * 3):
        now = time.time()
        idx = max(range(len(clients)), key=lambda i: now - key_last_call[i])
        elapsed = now - key_last_call[idx]
        if elapsed < CALL_GAP:
            time.sleep(CALL_GAP - elapsed)
        key_last_call[idx] = time.time()
        try:
            resp = _generate_with_timeout(
                clients[idx], contents, config, timeout=CALL_TIMEOUT)
            return resp.text.strip() if resp and resp.text else ''
        except TimeoutError as e:
            print(f"      Gemma timeout (attempt {attempt+1}): {e}", flush=True)
            time.sleep(3.0)
        except Exception as e:
            err = str(e)
            wait = 6.0 if ('quota' in err.lower() or '429' in err) else 3.0
            print(f"      Gemma error (attempt {attempt+1}): {err[:60]}", flush=True)
            time.sleep(wait)
    return ''


def build_prompt_contents(
    instruction: str,
    road_network: str,
    traj_descriptions: list,
    probs: np.ndarray,
    ade_values: np.ndarray,
    ranking: list,
    image_bytes_list: list,
):
    """Build the teacher contents (text + images) AND the student-facing text.

    Teacher prompt: includes the GT-derived ranking + per-trajectory ADE so
    Gemma can produce accurate reasoning, with explicit instructions to write
    in the student's voice (no GT references).

    Student prompt: identical inputs minus the ranking block — this is the
    text the finetuned VLM will see at inference time, stored for SFT.
    """
    traj_block = "\n".join(
        f"  #{i+1:>2} (conf={probs[i]*100:.1f}%): {traj_descriptions[i]}"
        for i in range(len(traj_descriptions))
    )
    ranking_block = "\n".join(
        f"  Rank {pos+1:>2}: #{traj_idx+1:>2}  (ADE={ade_values[traj_idx]:.2f}m)  "
        f"{traj_descriptions[traj_idx]}"
        for pos, traj_idx in enumerate(ranking)
    )

    road_network_block = (
        f"ROAD NETWORK AHEAD:\n{road_network}"
        if not road_network.startswith('(') else road_network
    )

    teacher_prompt = TEACHER_REASONING_PROMPT.format(
        instruction=instruction,
        road_network_block=road_network_block,
        traj_block=traj_block,
        ranking_block=ranking_block,
    )
    student_prompt = STUDENT_REASONING_PROMPT.format(
        instruction=instruction,
        road_network_block=road_network_block,
        traj_block=traj_block,
    )

    contents = []
    for img_bytes in image_bytes_list:
        if img_bytes:
            contents.append(gtypes.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'))
    contents.append(gtypes.Part.from_text(text=teacher_prompt))
    return contents, teacher_prompt, student_prompt


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--high_ade_only', action='store_true',
                        help='Process only the 19 high-ADE scenes')
    parser.add_argument('--out_dir', default='vlm_dataset',
                        help='Output directory for JSONL files')
    parser.add_argument('--resume', action='store_true',
                        help='Skip scenes already in out_dir/records.jsonl')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'records.jsonl')

    # Load already-completed records for resume.
    # Key is (sample_token, instruction) so different windows in the same scene
    # are tracked independently — required for the per-window loop below.
    done_keys = set()
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                done_keys.add((r.get('sample_token', ''), r['instruction']))
        print(f"Resume: {len(done_keys)} records already done.", flush=True)

    # ── Load PGP model ──
    print("Loading PGP model ...", flush=True)
    with open(CFG_FILE) as f:
        cfg = yaml.safe_load(f)
    model = load_model(cfg)
    print(f"  Model loaded on {device}", flush=True)

    # ── Load nuScenes (metadata only) ──
    print("Loading nuScenes metadata ...", flush=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=DATA_ROOT, verbose=False)
    print(f"  {len(nusc.scene)} scenes loaded.", flush=True)

    # Build scene_num -> scene_token map
    sc_name_to_token = {s['name']: s['token'] for s in nusc.scene}

    # ── Load annotations ──
    df = pd.read_csv(ANNOT_CSV).dropna(subset=['Instruction'])
    df['scene_num'] = df['Scene Number'].astype(float).astype(int)
    df['ann_type']  = df['Instruction Type'].fillna('?')

    # ── Build scene -> pickle map ──
    print("Building scene → pickle map ...", flush=True)
    scene_pkl_map = build_scene_pickle_map(PREPROC_DIR)
    print(f"  {len(scene_pkl_map)} val scenes with pickles.", flush=True)

    # ── Select scenes to process ──
    ann_scenes = sorted(df['scene_num'].unique())
    if args.high_ade_only:
        target_scenes = [s for s in HIGH_ADE if s in set(ann_scenes) and s in scene_pkl_map]
    else:
        target_scenes = [s for s in ann_scenes if s in scene_pkl_map]

    print(f"Target scenes: {len(target_scenes)}", flush=True)

    # ── Find working Gemini keys ──
    print("\nFinding working Gemini keys ...", flush=True)
    working_keys = find_working_keys(KEYS_FILE)
    if not working_keys:
        print("ERROR: No working Gemini keys.", flush=True)
        sys.exit(1)
    clients = [genai.Client(api_key=k) for k in working_keys]
    key_last_call = [0.0] * len(clients)
    print(f"  {len(clients)} working keys.", flush=True)

    # ── Process each (scene, window) pair ──
    # Every preprocessed window becomes its own unit of work — we no longer
    # collapse a scene down to the single "worst" window.
    total_written = 0
    total_leaks   = 0
    high_ade_set = set(HIGH_ADE)
    for sc_num_raw in target_scenes:
        sc_num = int(sc_num_raw)   # df['scene_num'].unique() yields np.int64; JSON needs Python int
        sc_name = f"scene-{sc_num:04d}"
        scene_token = sc_name_to_token.get(sc_name)
        if not scene_token:
            print(f"\n[scene-{sc_num:04d}] WARNING: no nuScenes scene token", flush=True)
            continue

        annotations = df[df['scene_num'] == sc_num]
        pickles = scene_pkl_map.get(sc_num, [])
        if not pickles:
            print(f"\n[scene-{sc_num:04d}] WARNING: no pickles", flush=True)
            continue

        print(f"\n[scene-{sc_num:04d}] {len(pickles)} window(s), "
              f"{len(annotations)} instruction(s)", flush=True)

        for pkl_path in pickles:
            try:
                trajs, probs, gt, sample_token = run_inference(model, pkl_path)
            except Exception as e:
                print(f"  inference failed on {pkl_path}: {e}", flush=True)
                continue

            ades       = ade_k(trajs, gt)
            top_k      = int(np.argmax(probs))
            ranking    = [int(x) for x in np.argsort(ades)]   # best -> worst, py ints
            traj_descs = [describe_trajectory(t) for t in trajs]
            gt_ranking_1idx = [r + 1 for r in ranking]

            print(f"  [{os.path.basename(pkl_path)}] "
                  f"top_conf_ADE={float(ades[top_k]):.2f}m  "
                  f"minADE={float(ades.min()):.2f}m  "
                  f"best=#{ranking[0]+1} worst=#{ranking[-1]+1}", flush=True)

            # Pre-build per-window context shared across instructions
            road_network = build_road_network_text(nusc, sample_token, scene_token)
            has_road_net = not road_network.startswith('(')
            cam_paths = get_cam_front_images(nusc, sample_token, n_frames=4)
            image_bytes_list = [load_image_bytes(p) for p in cam_paths]
            n_images = sum(1 for b in image_bytes_list if b is not None)
            print(f"    road_net={'OK' if has_road_net else 'NO'}, "
                  f"images={n_images}/{len(cam_paths)}", flush=True)

            for _, row in annotations.iterrows():
                instruction = str(row['Instruction']).strip()
                ann_type    = str(row['ann_type'])
                key = (sample_token, instruction)

                if key in done_keys:
                    print(f"    [skip] {instruction[:50]}", flush=True)
                    continue

                print(f"    → [{ann_type}] {instruction[:65]} ...", flush=True)

                contents, teacher_prompt, student_prompt = build_prompt_contents(
                    instruction=instruction,
                    road_network=road_network,
                    traj_descriptions=traj_descs,
                    probs=probs,
                    ade_values=ades,
                    ranking=ranking,
                    image_bytes_list=image_bytes_list,
                )

                reasoning = call_gemma(clients, key_last_call, contents, max_tokens=1024)
                used_images = n_images > 0
                if not reasoning and n_images > 0:
                    print(f"      Multimodal call failed — retrying text-only ...",
                          flush=True)
                    text_only = [gtypes.Part.from_text(text=teacher_prompt)]
                    reasoning = call_gemma(clients, key_last_call, text_only, max_tokens=1024)
                    used_images = False
                if not reasoning:
                    print(f"      WARNING: empty response, skipping.", flush=True)
                    continue

                gt_leak_terms = detect_gt_leak(reasoning)
                gt_leak_flag  = bool(gt_leak_terms)
                if gt_leak_flag:
                    total_leaks += 1

                record = {
                    'scene_num':       sc_num,
                    'instruction':     instruction,
                    'ann_type':        ann_type,
                    'sample_token':    sample_token,
                    'pkl_path':        pkl_path,
                    'is_high_ade':     bool(sc_num in high_ade_set),
                    'road_network':    road_network,
                    'trajectories': [
                        {
                            'idx':         i + 1,
                            'ade':         float(ades[i]),
                            'prob':        float(probs[i]),
                            'gt_rank':     ranking.index(i) + 1,
                            'description': traj_descs[i],
                            'traj':        trajs[i].tolist(),
                        }
                        for i in range(len(trajs))
                    ],
                    'gt_traj':         gt.tolist(),
                    'gt_ranking':      gt_ranking_1idx,
                    'top_conf_idx':    top_k + 1,
                    'top_conf_ade':    float(ades[top_k]),
                    'min_ade':         float(ades.min()),
                    'cam_front_paths': cam_paths,
                    'n_images_used':   n_images if used_images else 0,
                    'student_prompt':  student_prompt,
                    'teacher_prompt':  teacher_prompt,
                    'gemma_reasoning': reasoning,
                    'gt_leak_flag':    gt_leak_flag,
                    'gt_leak_terms':   gt_leak_terms,
                }

                with open(out_path, 'a') as f:
                    f.write(json.dumps(record) + '\n')
                done_keys.add(key)
                total_written += 1

                leak_tag = f" LEAK={gt_leak_terms}" if gt_leak_flag else ""
                print(f"      ✓ reasoning={len(reasoning)}c{leak_tag} → {out_path}",
                      flush=True)

    print(f"\nDone. {total_written} records written to {out_path} "
          f"({total_leaks} flagged for GT leakage).", flush=True)


if __name__ == '__main__':
    main()
