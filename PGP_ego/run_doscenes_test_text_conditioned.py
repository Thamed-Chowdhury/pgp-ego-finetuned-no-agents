"""
Test-split evaluation for the text-conditioned lvm_ranked_text model.

Mirrors run_doscenes_test_baseline.py exactly, except:
  - Loads the gemini-embedding-001 cache (built by embed_doscenes_gemini.py).
  - Builds an anchor_token -> text_embedding lookup using the test split's
    scene names (which map to scene numbers in the doScenes annotation CSVs).
  - At inference time, attaches inputs['text_embedding'] to every batch element
    before calling the model.
  - Loads the lvm_ranked_text checkpoint (which has the extra text_proj layer).
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
    nusc = NuScenes(version='v1.0-test', dataroot=test_root, verbose=False)
    return PredictHelper(nusc)


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


def build_text_lookup(helper, scene_tuples, emb_pkl):
    with open(emb_pkl, 'rb') as f:
        cache = pickle.load(f)
    instr_emb = cache['instr_embeddings']
    scene_to_instr = cache['scene_to_instruction']
    empty_emb = cache['empty_embedding']
    text_dim = int(cache['dim'])

    nusc = helper.data
    lookup = {}
    n_hit = n_miss = 0
    for anchor_tok, scene_tok in scene_tuples:
        scene = nusc.get('scene', scene_tok)
        try:
            scene_num = int(scene['name'].split('-')[1])
        except Exception:
            scene_num = -1
        instr = scene_to_instr.get(scene_num, '')
        if instr and instr in instr_emb:
            lookup[anchor_tok] = instr_emb[instr]
            n_hit += 1
        else:
            lookup[anchor_tok] = empty_emb
            n_miss += 1
    print(f'[text-cache] {n_hit} anchors with annotation, {n_miss} without (zero-emb fallback)')
    return lookup, text_dim


def stage_preprocess(cfg, helper, test_dir, trainval_stats, batch_size=4, num_workers=0):
    os.makedirs(test_dir, exist_ok=True)
    stats_target = os.path.join(test_dir, 'stats.pickle')
    if not os.path.isfile(stats_target):
        if not os.path.isfile(trainval_stats):
            raise FileNotFoundError(f'trainval stats not found: {trainval_stats}')
        shutil.copy(trainval_stats, stats_target)

    sa = dict(cfg['test_set_args'])
    sa['random_flips'] = False
    sa['split'] = 'doscenes_test'
    # The doScenes class doesn't read text_emb_pkl; remove if present.
    sa.pop('text_emb_pkl', None)

    ds = NuScenesEgoGraphsDoScenes('extract_data', test_dir, sa, helper)
    expected = {t + '.pickle' for t in ds.token_list}
    existing = {f for f in os.listdir(test_dir) if f.startswith('ego_') and f.endswith('.pickle')}
    missing = expected - existing
    if not missing:
        print(f'[preprocess] all {len(expected)} pickles present in {test_dir}, skipping.')
        return
    print(f'[preprocess] extracting {len(missing)} missing pickles ...')
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for i, _ in enumerate(dl):
        if (i + 1) % 5 == 0 or (i + 1) == len(dl):
            print(f'  extract batch {i+1}/{len(dl)}')


def stage_inference(cfg, helper, test_dir, checkpoint, text_lookup, text_dim,
                    batch_size=4, num_workers=0, num_samples=100):
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
    print(f'[inference] ckpt: {checkpoint}  (val_metric={ckpt.get("val_metric", "?")})')

    sa = dict(cfg['test_set_args'])
    sa['random_flips'] = False
    sa['split'] = 'doscenes_test'
    sa.pop('text_emb_pkl', None)

    ds = NuScenesEgoGraphsDoScenes('load_data', test_dir, sa, helper)
    anchor_tokens = [t.split('_', 1)[1] for t in ds.token_list]
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    empty = np.zeros(text_dim, dtype=np.float32)
    out = {}
    cursor = 0
    with torch.no_grad():
        for bi, data in enumerate(dl):
            data = u.send_to_device(u.convert_double_to_float(data))
            # Attach text embeddings for this batch.
            batch_tokens = anchor_tokens[cursor:cursor + data['ground_truth']['traj'].shape[0]]
            embs = np.stack([text_lookup.get(t, empty) for t in batch_tokens], axis=0)
            data['inputs']['text_embedding'] = torch.as_tensor(embs, dtype=torch.float32, device=device)

            preds = model(data['inputs'])
            traj  = preds['traj'].detach().cpu().numpy()
            probs = preds['probs'].detach().cpu().numpy()
            gt    = data['ground_truth']['traj'].detach().cpu().numpy()
            for b in range(traj.shape[0]):
                top_idx = int(np.argmax(probs[b]))
                out[batch_tokens[b]] = {
                    'top_idx':      top_idx,
                    'top_traj_pgp': traj[b, top_idx].astype(np.float32),
                    'all_traj_pgp': traj[b].astype(np.float32),
                    'probs':        probs[b].astype(np.float32),
                    'gt_pgp':       gt[b].astype(np.float32),
                }
            cursor += traj.shape[0]
            if (bi + 1) % 5 == 0 or cursor == len(ds):
                print(f'  infer batch {bi+1}/{len(dl)}  ({cursor}/{len(ds)})')
    return out


def stage_submission(out, scene_tuples, out_csv, scene_to_instr=None):
    """Write submission.csv in the official doScenes format:
        sample_token, instruction, x1, y1, ..., x12, y12
    `scene_to_instr` maps scene_token -> instruction string the pipeline
    actually consumed (empty for unannotated scenes, and empty for every
    row in the no-language ablation). If None, all instructions are empty."""
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    header = ['sample_token', 'instruction']
    for i in range(1, FUTURE_LEN + 1):
        header += [f'x{i}', f'y{i}']
    rows, missing = [], []
    for anchor_tok, scene_tok in scene_tuples:
        if anchor_tok not in out:
            missing.append((anchor_tok, scene_tok)); continue
        chal = pgp_to_challenge_frame(out[anchor_tok]['top_traj_pgp'])
        instr = '' if scene_to_instr is None else scene_to_instr.get(scene_tok, '')
        row = [scene_tok, instr]
        for x, y in chal:
            row.extend([f'{float(x):.6f}', f'{float(y):.6f}'])
        rows.append(row)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f'[submission] wrote {len(rows)} rows to {out_csv}')
    return rows, missing


def stage_self_eval(out, scene_tuples, helper, test_root, doscenes_repo, out_json):
    import importlib.util
    metrics_path = os.path.join(doscenes_repo, 'metrics.py')
    spec = importlib.util.spec_from_file_location('doscenes_metrics', metrics_path)
    doscenes_metrics = importlib.util.module_from_spec(spec); spec.loader.exec_module(doscenes_metrics)
    compute_ego_metrics = doscenes_metrics.compute_ego_metrics
    from nuscenes.map_expansion.map_api import NuScenesMap
    nusc = helper.data
    map_cache = {}
    def loc(sc_t):
        sc = nusc.get('scene', sc_t); lg = nusc.get('log', sc['log_token']); return lg['location']
    per_scene = []
    for anchor_tok, scene_tok in scene_tuples:
        if anchor_tok not in out: continue
        location = loc(scene_tok)
        if location not in map_cache:
            print(f'[self-eval] loading map: {location}')
            map_cache[location] = NuScenesMap(dataroot=test_root, map_name=location)
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
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({'aggregate': agg, 'per_scene': per_scene, 'n_scenes': len(per_scene)}, f, indent=2)
    print(f'[self-eval] wrote {out_json}')
    return agg, per_scene


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',          required=True)
    p.add_argument('--test_root',       required=True)
    p.add_argument('--trainval_stats',  required=True)
    p.add_argument('--test_preproc',    required=True)
    p.add_argument('--checkpoint',      required=True)
    p.add_argument('--doscenes_repo',   required=True)
    p.add_argument('--text_emb_pkl',    required=True)
    p.add_argument('--out_dir',         required=True)
    p.add_argument('--batch_size',      type=int, default=4)
    p.add_argument('--num_workers',     type=int, default=0)
    p.add_argument('--num_samples',     type=int, default=100)
    p.add_argument('--skip_preprocess', action='store_true')
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg['version'] = 'v1.0-test'

    helper = make_test_helper(args.test_root)
    nusc = helper.data
    print(f'[main] {len(nusc.scene)} test scenes loaded')
    scene_tuples = scenes_used_for_eval(nusc)
    print(f'[main] {len(scene_tuples)} scenes meet >= {MIN_SCENE_SAMPLES} samples')

    text_lookup, text_dim = build_text_lookup(helper, scene_tuples, args.text_emb_pkl)

    if not args.skip_preprocess:
        stage_preprocess(cfg, helper, args.test_preproc, args.trainval_stats,
                         batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        print('[main] --skip_preprocess')

    out = stage_inference(cfg, helper, args.test_preproc, args.checkpoint,
                          text_lookup, text_dim,
                          batch_size=args.batch_size, num_workers=args.num_workers,
                          num_samples=args.num_samples)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'inference_cache.pkl'), 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Build scene_token -> instruction map so the submission CSV records the
    # exact text the model consumed (instruction column is part of the
    # doScenes submission spec).
    nusc = helper.data
    scene_to_instr = {}
    with open(args.text_emb_pkl, 'rb') as f:
        emb_cache = pickle.load(f)
    s2i = emb_cache['scene_to_instruction']
    for anchor_tok, scene_tok in scene_tuples:
        scene = nusc.get('scene', scene_tok)
        try:
            scene_num = int(scene['name'].split('-')[1])
        except Exception:
            scene_num = -1
        scene_to_instr[scene_tok] = s2i.get(scene_num, '')
    stage_submission(out, scene_tuples, os.path.join(args.out_dir, 'submission.csv'),
                     scene_to_instr=scene_to_instr)
    agg, _ = stage_self_eval(out, scene_tuples, helper, args.test_root,
                             args.doscenes_repo, os.path.join(args.out_dir, 'self_eval_metrics.json'))

    print('\n========== Test split results (text-conditioned) ==========')
    print(f'   N scenes evaluated: {len([t for t in scene_tuples if t[0] in out])}')
    for k in ['ade_2s', 'ade_4s', 'ade_6s', 'fde', 'miss_rate',
              'speed_error', 'ahe', 'fhe', 'offroad', 'offroad_rate', 'offyaw']:
        print(f'   {k:14s} = {agg[k]:.4f}')
    print('============================================================')


if __name__ == '__main__':
    main()
