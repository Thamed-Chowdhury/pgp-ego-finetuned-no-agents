"""
End-to-end pipeline for the doScenes Instructed Driving Challenge test split,
*baseline track* (top-confidence PGP trajectory, no language conditioning).

Steps:
  1. Set up nuScenes v1.0-test + a doScenes-style ego dataset (one window per
     scene, anchor at sample index 4).
  2. Re-use trainval stats (upper bound) to skip compute_stats. Run extract_data
     to cache per-window pickles into TEST_PREPROCESSED_DIR.
  3. Load the trained PGP checkpoint and run forward inference. For each
     window, take the highest-probability cluster trajectory.
  4. Convert from PGP local frame (+y forward, +x right) to the challenge local
     frame (+x forward, +y left) and write submission.csv keyed by scene_token.
  5. Self-evaluate using doScenes_repo/metrics.py (per-step ADE/FDE in local
     frame, plus map-aware off-road / off-yaw using the test split's maps).

The whole flow runs in this single script so that there is no manual coupling
between preprocessing, inference, and submission generation.
"""

import argparse
import csv
import json
import os
import shutil
import sys
import warnings

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# --- Monkey-patch get_prediction_challenge_split BEFORE importing the dataset.
# The parent NuScenesTrajectories.__init__ calls this on every dataset
# instantiation. The doScenes test split is not in the prediction-challenge
# split files, so the call would fail. We replace it with a no-op (the
# returned token_list gets overwritten by our subclass anyway).
import nuscenes.eval.prediction.splits as _ns_pred_splits
_ns_pred_splits.get_prediction_challenge_split = lambda split, dataroot=None: []

from datasets.nuScenes.nuScenes_ego_graphs_doscenes import (
    NuScenesEgoGraphsDoScenes, HISTORY_LEN, FUTURE_LEN, MIN_SCENE_SAMPLES,
)
from train_eval.initialization import initialize_prediction_model
import train_eval.utils as u

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
DEFAULT_TEST_ROOT      = r'D:\DriveX_PGP\nuScenes_data\v1-test'
DEFAULT_TRAINVAL_STATS = r'D:\DriveX_PGP\pgp-ego-prediction\pgp_ego_preprocessed\stats.pickle'
DEFAULT_TEST_PREPROC   = r'D:\DriveX_PGP\pgp-ego-prediction\pgp_ego_test_preprocessed'
DEFAULT_CHECKPOINT     = r'D:\DriveX_PGP\pgp-ego-prediction\pgp_ego_output2\checkpoints\best.tar'
DEFAULT_CONFIG         = os.path.join(HERE, 'configs', 'pgp_ego_gatx2_lvm_traversal.yml')
DEFAULT_DOSCENES_REPO  = r'D:\DriveX_PGP\doScenes_repo'
DEFAULT_OUT_DIR        = r'D:\DriveX_PGP\test_baseline_results'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_test_helper(test_root):
    from nuscenes import NuScenes
    from nuscenes.prediction import PredictHelper
    nusc = NuScenes(version='v1.0-test', dataroot=test_root, verbose=False)
    helper = PredictHelper(nusc)
    return helper


def scenes_used_for_eval(nusc):
    """Replicate the dataset's window selection so we can map anchor -> scene_token."""
    out = []  # list of (anchor_sample_token, scene_token)
    for s in sorted(nusc.scene, key=lambda x: x['name']):
        samples = []
        t = s['first_sample_token']
        while t:
            samples.append(t)
            t = nusc.get('sample', t)['next']
        if len(samples) >= MIN_SCENE_SAMPLES:
            out.append((samples[HISTORY_LEN], s['token']))
    return out


def pgp_to_challenge_frame(traj_pgp):
    """
    PGP local: +y forward, +x right    →    challenge local: +x forward, +y left.
    For a (..., 2) array: (cx, cy) = (py, -px).
    """
    out = np.empty_like(traj_pgp)
    out[..., 0] = traj_pgp[..., 1]
    out[..., 1] = -traj_pgp[..., 0]
    return out


def get_anchor_world_pose(nusc, anchor_token):
    """Returns (anchor_pos_xy, anchor_yaw) in world coords, via LIDAR_TOP."""
    from pyquaternion import Quaternion
    from nuscenes.eval.common.utils import quaternion_yaw
    sample = nusc.get('sample', anchor_token)
    ld_tok = sample['data']['LIDAR_TOP']
    ld     = nusc.get('sample_data', ld_tok)
    ep     = nusc.get('ego_pose', ld['ego_pose_token'])
    pos = np.array(ep['translation'][:2], dtype=np.float64)
    yaw = float(quaternion_yaw(Quaternion(ep['rotation'])))
    return pos, yaw


def get_future_world_xy(nusc, anchor_token, n_future=FUTURE_LEN):
    """Return (n_future, 2) array of future ego world positions starting from
    the sample AFTER anchor (consistent with the doScenes dataloader)."""
    cur = nusc.get('sample', anchor_token)
    out = []
    for _ in range(n_future):
        cur = nusc.get('sample', cur['next'])
        ld = nusc.get('sample_data', cur['data']['LIDAR_TOP'])
        ep = nusc.get('ego_pose', ld['ego_pose_token'])
        out.append(ep['translation'][:2])
    return np.array(out, dtype=np.float64)


def world_to_challenge_local(world_xy, anchor_pos, anchor_yaw):
    """
    Inverse of metrics.compute_ego_metrics's R: pred_world = R @ pred_local + anchor_pos
    where R = [[cos y, -sin y], [sin y, cos y]]  (anchor_yaw is the world heading
    of the local +x axis). So local = R^T @ (world - anchor_pos).
    """
    cos_y, sin_y = np.cos(anchor_yaw), np.sin(anchor_yaw)
    R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    return (R.T @ (world_xy - anchor_pos).T).T


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def stage_preprocess(cfg, helper, test_dir, trainval_stats, batch_size=4, num_workers=0):
    """Extract + save per-window pickles into test_dir. Reuses trainval stats."""
    os.makedirs(test_dir, exist_ok=True)
    stats_target = os.path.join(test_dir, 'stats.pickle')
    if not os.path.isfile(stats_target):
        if not os.path.isfile(trainval_stats):
            raise FileNotFoundError(
                f'trainval stats not found at {trainval_stats}; cannot reuse')
        shutil.copy(trainval_stats, stats_target)
        print(f'[preprocess] Copied stats from {trainval_stats} -> {stats_target}')

    sa = dict(cfg['test_set_args'])
    sa['random_flips'] = False
    sa['split'] = 'doscenes_test'  # ignored by our subclass; kept for arg dict completeness

    ds = NuScenesEgoGraphsDoScenes('extract_data', test_dir, sa, helper)
    expected = {t + '.pickle' for t in ds.token_list}
    existing = {f for f in os.listdir(test_dir) if f.startswith('ego_') and f.endswith('.pickle')}
    missing = expected - existing
    if not missing:
        print(f'[preprocess] all {len(expected)} pickles already present in {test_dir}, skipping extract.')
        return

    if existing:
        print(f'[preprocess] {len(existing)} pickles already present, '
              f'extracting {len(missing)} missing ones.')
    print(f'[preprocess] {len(ds)} windows total, extracting...')
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for i, _ in enumerate(dl):
        if (i + 1) % 5 == 0 or (i + 1) == len(dl):
            print(f'  extract batch {i+1}/{len(dl)}')


def stage_inference(cfg, helper, test_dir, checkpoint, batch_size=4,
                    num_workers=0, num_samples=100):
    """Load checkpoint, run forward, return dict[anchor_sample_token] -> result."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'[inference] device = {device}')

    model = initialize_prediction_model(
        cfg['encoder_type'], cfg['aggregator_type'], cfg['decoder_type'],
        cfg['encoder_args'], cfg['aggregator_args'], cfg['decoder_args'],
    ).float().to(device)
    model.eval()
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.aggregator.num_samples = num_samples
    model.decoder.num_samples = num_samples
    print(f'[inference] checkpoint loaded: {checkpoint}')

    sa = dict(cfg['test_set_args'])
    sa['random_flips'] = False
    sa['split'] = 'doscenes_test'

    ds = NuScenesEgoGraphsDoScenes('load_data', test_dir, sa, helper)
    tokens = [t.split('_', 1)[1] for t in ds.token_list]
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    out = {}
    cursor = 0
    with torch.no_grad():
        for bi, data in enumerate(dl):
            data = u.send_to_device(u.convert_double_to_float(data))
            preds = model(data['inputs'])
            traj  = preds['traj'].detach().cpu().numpy()    # (B, K, T, 2)
            probs = preds['probs'].detach().cpu().numpy()   # (B, K)
            gt    = data['ground_truth']['traj'].detach().cpu().numpy()  # (B, T, 2)

            for b in range(traj.shape[0]):
                top_idx = int(np.argmax(probs[b]))
                out[tokens[cursor + b]] = {
                    'top_idx':       top_idx,
                    'top_traj_pgp':  traj[b, top_idx].astype(np.float32),
                    'all_traj_pgp':  traj[b].astype(np.float32),
                    'probs':         probs[b].astype(np.float32),
                    'gt_pgp':        gt[b].astype(np.float32),
                }
            cursor += traj.shape[0]
            if (bi + 1) % 5 == 0 or cursor == len(ds):
                print(f'  infer batch {bi+1}/{len(dl)}  ({cursor}/{len(ds)})')
    return out


def stage_submission(out, scene_tuples, out_csv):
    """Write submission.csv keyed by scene_token, in challenge local frame."""
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    header = ['sample_token']
    for i in range(1, FUTURE_LEN + 1):
        header += [f'x{i}', f'y{i}']

    rows = []
    missing = []
    for anchor_tok, scene_tok in scene_tuples:
        if anchor_tok not in out:
            missing.append((anchor_tok, scene_tok))
            continue
        chal = pgp_to_challenge_frame(out[anchor_tok]['top_traj_pgp'])  # (12, 2)
        row = [scene_tok]
        for x, y in chal:
            row.extend([f'{float(x):.6f}', f'{float(y):.6f}'])
        rows.append(row)

    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f'[submission] wrote {len(rows)} rows to {out_csv}')
    if missing:
        print(f'[submission] WARNING: {len(missing)} anchors had no inference result')
    return rows, missing


def stage_self_eval(out, scene_tuples, helper, test_root, doscenes_repo, out_json):
    """Per-scene + aggregate metrics using the official compute_ego_metrics()."""
    # Load doScenes_repo/metrics.py by file path to avoid name conflict with
    # the local PGP_ego/metrics/ package.
    import importlib.util
    metrics_path = os.path.join(doscenes_repo, 'metrics.py')
    spec = importlib.util.spec_from_file_location('doscenes_metrics', metrics_path)
    doscenes_metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(doscenes_metrics)
    compute_ego_metrics = doscenes_metrics.compute_ego_metrics
    from nuscenes.map_expansion.map_api import NuScenesMap

    nusc = helper.data
    map_cache = {}

    # Build per-scene location lookup
    def get_location(scene_token):
        scene = nusc.get('scene', scene_token)
        log = nusc.get('log', scene['log_token'])
        return log['location']

    per_scene = []
    for anchor_tok, scene_tok in scene_tuples:
        if anchor_tok not in out:
            continue

        location = get_location(scene_tok)
        if location not in map_cache:
            print(f'[self-eval] loading map: {location}')
            map_cache[location] = NuScenesMap(dataroot=test_root, map_name=location)
        nusc_map = map_cache[location]

        anchor_pos, anchor_yaw = get_anchor_world_pose(nusc, anchor_tok)
        future_world = get_future_world_xy(nusc, anchor_tok, FUTURE_LEN)
        future_local = world_to_challenge_local(future_world, anchor_pos, anchor_yaw)

        pred_local = pgp_to_challenge_frame(out[anchor_tok]['top_traj_pgp'])

        m = compute_ego_metrics(pred_local, future_local, anchor_pos, anchor_yaw, nusc_map)
        per_scene.append({
            'scene_token':   scene_tok,
            'anchor_token':  anchor_tok,
            'metrics':       m,
        })

    # Aggregate
    keys = ['ade_2s', 'ade_4s', 'ade_6s', 'fde', 'miss_rate',
            'speed_error', 'ahe', 'fhe', 'offroad', 'offroad_rate', 'offyaw']
    agg = {k: float(np.mean([r['metrics'][k] for r in per_scene])) for k in keys}

    # Save
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({'aggregate': agg, 'per_scene': per_scene, 'n_scenes': len(per_scene)},
                  f, indent=2)
    print(f'[self-eval] wrote {out_json}')

    return agg, per_scene


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',          default=DEFAULT_CONFIG)
    p.add_argument('--test_root',       default=DEFAULT_TEST_ROOT)
    p.add_argument('--trainval_stats',  default=DEFAULT_TRAINVAL_STATS)
    p.add_argument('--test_preproc',    default=DEFAULT_TEST_PREPROC)
    p.add_argument('--checkpoint',      default=DEFAULT_CHECKPOINT)
    p.add_argument('--doscenes_repo',   default=DEFAULT_DOSCENES_REPO)
    p.add_argument('--out_dir',         default=DEFAULT_OUT_DIR)
    p.add_argument('--batch_size',      type=int, default=4)
    p.add_argument('--num_workers',     type=int, default=0)
    p.add_argument('--num_samples',     type=int, default=100)
    p.add_argument('--skip_preprocess', action='store_true')
    p.add_argument('--skip_inference',  action='store_true')
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg['version'] = 'v1.0-test'

    helper = make_test_helper(args.test_root)
    nusc = helper.data
    print(f'[main] {len(nusc.scene)} test scenes loaded')

    scene_tuples = scenes_used_for_eval(nusc)
    print(f'[main] {len(scene_tuples)} scenes meet >= {MIN_SCENE_SAMPLES} samples')

    if not args.skip_preprocess:
        stage_preprocess(cfg, helper, args.test_preproc, args.trainval_stats,
                         batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        print('[main] skipping preprocessing per --skip_preprocess')

    inf_cache = os.path.join(args.out_dir, 'inference_cache.pkl')
    if args.skip_inference and os.path.isfile(inf_cache):
        import pickle
        with open(inf_cache, 'rb') as f:
            out = pickle.load(f)
        print(f'[main] loaded cached inference: {len(out)} entries')
    else:
        out = stage_inference(cfg, helper, args.test_preproc, args.checkpoint,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              num_samples=args.num_samples)
        os.makedirs(args.out_dir, exist_ok=True)
        import pickle
        with open(inf_cache, 'wb') as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'[main] cached inference -> {inf_cache}')

    submission_csv = os.path.join(args.out_dir, 'submission.csv')
    stage_submission(out, scene_tuples, submission_csv)

    metrics_json = os.path.join(args.out_dir, 'self_eval_metrics.json')
    agg, _ = stage_self_eval(out, scene_tuples, helper, args.test_root,
                             args.doscenes_repo, metrics_json)

    print('\n=========== Test split results (top-conf baseline, no instruction) ===========')
    print(f'   N scenes evaluated: {len([t for t in scene_tuples if t[0] in out])}')
    for k in ['ade_2s', 'ade_4s', 'ade_6s', 'fde', 'miss_rate',
              'speed_error', 'ahe', 'fhe', 'offroad', 'offroad_rate', 'offyaw']:
        print(f'   {k:14s} = {agg[k]:.4f}')
    print('================================================================================')


if __name__ == '__main__':
    main()
