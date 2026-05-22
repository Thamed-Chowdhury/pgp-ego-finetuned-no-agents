"""
Ablation companion to run_doscenes_test_text_conditioned.py: same model,
same test pickles, but **all text embeddings forced to zero**.

This isolates the contribution of the doScenes instruction at inference time
for the doScenes Ablation track: re-running with zero text gives the
"without language" number while keeping every other variable identical.
"""

import argparse
import csv
import json
import os
import pickle
import shutil
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import nuscenes.eval.prediction.splits as _ns_pred_splits
_ns_pred_splits.get_prediction_challenge_split = lambda split, dataroot=None: []

from datasets.nuScenes.nuScenes_ego_graphs_doscenes import (
    NuScenesEgoGraphsDoScenes, HISTORY_LEN, FUTURE_LEN, MIN_SCENE_SAMPLES,
)
from train_eval.initialization import initialize_prediction_model
import train_eval.utils as u

from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw


def make_test_helper(test_root):
    from nuscenes import NuScenes
    from nuscenes.prediction import PredictHelper
    return PredictHelper(NuScenes(version='v1.0-test', dataroot=test_root, verbose=False))


def scenes_used_for_eval(nusc):
    out = []
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
    out = np.empty_like(traj_pgp)
    out[..., 0] = traj_pgp[..., 1]
    out[..., 1] = -traj_pgp[..., 0]
    return out


def get_anchor_world_pose(nusc, anchor_token):
    sample = nusc.get('sample', anchor_token)
    ld = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ep = nusc.get('ego_pose', ld['ego_pose_token'])
    return np.array(ep['translation'][:2], dtype=np.float64), float(quaternion_yaw(Quaternion(ep['rotation'])))


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',          required=True)
    p.add_argument('--test_root',       required=True)
    p.add_argument('--trainval_stats',  required=True)
    p.add_argument('--test_preproc',    required=True)
    p.add_argument('--checkpoint',      required=True)
    p.add_argument('--doscenes_repo',   required=True)
    p.add_argument('--text_dim',        type=int, default=768)
    p.add_argument('--out_dir',         required=True)
    p.add_argument('--batch_size',      type=int, default=4)
    p.add_argument('--num_samples',     type=int, default=100)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg['version'] = 'v1.0-test'

    helper = make_test_helper(args.test_root)
    nusc = helper.data
    print(f'[main] {len(nusc.scene)} test scenes loaded')
    scene_tuples = scenes_used_for_eval(nusc)
    print(f'[main] {len(scene_tuples)} scenes meet >= {MIN_SCENE_SAMPLES} samples')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = initialize_prediction_model(
        cfg['encoder_type'], cfg['aggregator_type'], cfg['decoder_type'],
        cfg['encoder_args'], cfg['aggregator_args'], cfg['decoder_args'],
    ).float().to(device)
    model.eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.aggregator.num_samples = args.num_samples
    model.decoder.num_samples = args.num_samples
    print(f'[main] ckpt: {args.checkpoint} (val_metric={ckpt.get("val_metric", "?")})')

    sa = dict(cfg['test_set_args'])
    sa['random_flips'] = False
    sa['split'] = 'doscenes_test'
    sa.pop('text_emb_pkl', None)
    ds = NuScenesEgoGraphsDoScenes('load_data', args.test_preproc, sa, helper)
    anchor_tokens = [t.split('_', 1)[1] for t in ds.token_list]
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f'[main] {len(ds)} test anchors; forcing zero text embeddings (ablation)')

    out = {}
    cursor = 0
    with torch.no_grad():
        for bi, data in enumerate(dl):
            data = u.send_to_device(u.convert_double_to_float(data))
            B = data['ground_truth']['traj'].shape[0]
            zero_emb = torch.zeros(B, args.text_dim, dtype=torch.float32, device=device)
            data['inputs']['text_embedding'] = zero_emb
            preds = model(data['inputs'])
            traj  = preds['traj'].detach().cpu().numpy()
            probs = preds['probs'].detach().cpu().numpy()
            for b in range(B):
                top_idx = int(np.argmax(probs[b]))
                out[anchor_tokens[cursor + b]] = {
                    'top_idx': top_idx,
                    'top_traj_pgp': traj[b, top_idx].astype(np.float32),
                    'probs': probs[b].astype(np.float32),
                }
            cursor += B
            if (bi + 1) % 5 == 0 or cursor == len(ds):
                print(f'  infer batch {bi+1}/{len(dl)}  ({cursor}/{len(ds)})')

    os.makedirs(args.out_dir, exist_ok=True)
    # doScenes submission spec format:
    #     sample_token, instruction, x1, y1, ..., x12, y12
    # This is the no-language ablation, so the instruction column is empty
    # for every row.
    header = ['sample_token', 'instruction']
    for i in range(1, FUTURE_LEN + 1):
        header += [f'x{i}', f'y{i}']
    rows = []
    for anchor_tok, scene_tok in scene_tuples:
        if anchor_tok not in out:
            continue
        chal = pgp_to_challenge_frame(out[anchor_tok]['top_traj_pgp'])
        row = [scene_tok, '']
        for x, y in chal:
            row.extend([f'{float(x):.6f}', f'{float(y):.6f}'])
        rows.append(row)
    submission_csv = os.path.join(args.out_dir, 'submission.csv')
    with open(submission_csv, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f'[submit] wrote {submission_csv}  ({len(rows)} rows)')

    import importlib.util
    metrics_path = os.path.join(args.doscenes_repo, 'metrics.py')
    spec = importlib.util.spec_from_file_location('doscenes_metrics', metrics_path)
    doscenes_metrics = importlib.util.module_from_spec(spec); spec.loader.exec_module(doscenes_metrics)
    compute_ego_metrics = doscenes_metrics.compute_ego_metrics
    from nuscenes.map_expansion.map_api import NuScenesMap
    map_cache = {}
    def loc(sc_t):
        sc = nusc.get('scene', sc_t); lg = nusc.get('log', sc['log_token']); return lg['location']
    per_scene = []
    for anchor_tok, scene_tok in scene_tuples:
        if anchor_tok not in out: continue
        location = loc(scene_tok)
        if location not in map_cache:
            print(f'[self-eval] loading map: {location}')
            map_cache[location] = NuScenesMap(dataroot=args.test_root, map_name=location)
        nusc_map = map_cache[location]
        anchor_pos, anchor_yaw = get_anchor_world_pose(nusc, anchor_tok)
        future_world = get_future_world_xy(nusc, anchor_tok, FUTURE_LEN)
        future_local = world_to_challenge_local(future_world, anchor_pos, anchor_yaw)
        pred_local = pgp_to_challenge_frame(out[anchor_tok]['top_traj_pgp'])
        m = compute_ego_metrics(pred_local, future_local, anchor_pos, anchor_yaw, nusc_map)
        per_scene.append({'scene_token': scene_tok, 'anchor_token': anchor_tok, 'metrics': m})
    keys = ['ade_2s', 'ade_4s', 'ade_6s', 'fde', 'miss_rate',
            'speed_error', 'ahe', 'fhe', 'offroad', 'offroad_rate', 'offyaw']
    agg = {k: float(np.mean([r['metrics'][k] for r in per_scene])) for k in keys}
    with open(os.path.join(args.out_dir, 'self_eval_metrics.json'), 'w') as f:
        json.dump({'aggregate': agg, 'per_scene': per_scene, 'n_scenes': len(per_scene)}, f, indent=2)

    print('\n===== Without-language ablation (zero text embedding) =====')
    print(f'   N scenes evaluated: {len(per_scene)}')
    for k in keys:
        print(f'   {k:14s} = {agg[k]:.4f}')


if __name__ == '__main__':
    main()
