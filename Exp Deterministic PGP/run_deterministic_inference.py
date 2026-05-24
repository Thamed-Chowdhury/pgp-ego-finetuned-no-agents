"""
Deterministic test-split evaluation for the text-conditioned lvm_ranked_text
model.

Same flow as PGP_ego/run_doscenes_test_text_conditioned.py, with one purpose:
make the ADE@6s number bit-stable across reruns by seeding every RNG that the
PGP pipeline touches.

Sources of stochasticity in the pipeline (and how we kill each one):

  1. PGP aggregator -- Categorical(pi_s).sample() over policy traversals.
       -> torch.manual_seed + torch.cuda.manual_seed_all.
  2. LVMRankedText decoder -- torch.randn for latent z (1000 samples).
       -> same torch CUDA RNG.
  3. K-means over the 1000 trajectories (sklearn KMeans, init='random',
       n_init=1) inside ray workers.
       -> monkey-patch cluster_and_rank to pass random_state=seed.
       Ray worker numpy RNG is also seeded for belt-and-braces.
  4. cuDNN convolution/attention algorithm selection.
       -> torch.backends.cudnn.deterministic = True, benchmark = False.

This script also fixes the dataloader to single-process and shuffle=False so
batch order is stable. The ckpt is loaded once, num_samples is locked to the
value passed on the CLI (default 1000, matching the training config), and the
batch_size is held at 4 to match the original eval that produced 2.6526m.
"""

import argparse
import csv
import json
import os
import pickle
import random
import shutil
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Path wiring -- this script lives in `Exp Deterministic PGP/` and imports
# PGP_ego modules from the repo root.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
PGP_EGO_DIR = os.path.join(REPO_ROOT, 'PGP_ego')
sys.path.insert(0, PGP_EGO_DIR)

# Stub out the nuScenes prediction-challenge-split helper before any PGP_ego
# import pulls it in -- same trick the upstream test script uses.
import nuscenes.eval.prediction.splits as _ns_pred_splits  # noqa: E402
_ns_pred_splits.get_prediction_challenge_split = lambda split, dataroot=None: []


# ---------------------------------------------------------------------------
# Determinism patches. Applied at import time before models/utils.py finishes.
# ---------------------------------------------------------------------------
def patch_kmeans_determinism(seed):
    """Replace utils.cluster_and_rank with a version that forces a fixed
    random_state into KMeans and re-seeds numpy inside each ray worker."""
    import models.decoders.utils as decoder_utils
    from sklearn.cluster import KMeans
    import ray
    from scipy.spatial.distance import cdist

    @ray.remote
    def deterministic_cluster_and_rank(k, data):
        # Re-seed inside the ray worker so any np-random fallback is also fixed.
        np.random.seed(seed)

        def cluster(n_clusters, x):
            clustering_op = KMeans(
                n_clusters=n_clusters,
                n_init=1,
                max_iter=100,
                init='random',
                random_state=seed,
            ).fit(x)
            return clustering_op.labels_, clustering_op.cluster_centers_

        def rank_clusters(cluster_counts, cluster_centers):
            num_clusters = len(cluster_counts)
            cluster_ids = np.arange(num_clusters)
            ranks = np.ones(num_clusters)
            for i in range(num_clusters, 0, -1):
                centroid_dists = cdist(cluster_centers, cluster_centers)
                n1 = cluster_counts.reshape(1, -1).repeat(len(cluster_counts), axis=0)
                n2 = n1.transpose()
                wts = n1 * n2 / (n1 + n2)
                dists = wts * centroid_dists + np.diag(np.inf * np.ones(len(cluster_counts)))
                c1, c2 = np.unravel_index(dists.argmin(), dists.shape)
                c = c1 if cluster_counts[c1] <= cluster_counts[c2] else c2
                c_ = c2 if cluster_counts[c1] <= cluster_counts[c2] else c1
                ranks[cluster_ids[c]] = i
                cluster_centers[c_] = (cluster_counts[c_] * cluster_centers[c_]
                                       + cluster_counts[c] * cluster_centers[c]) \
                                      / (cluster_counts[c_] + cluster_counts[c])
                cluster_counts[c_] += cluster_counts[c]
                cluster_ids = np.delete(cluster_ids, c)
                cluster_centers = np.delete(cluster_centers, c, axis=0)
                cluster_counts = np.delete(cluster_counts, c)
            return ranks

        cluster_lbls, cluster_ctrs = cluster(k, data)
        cluster_cnts = np.unique(cluster_lbls, return_counts=True)[1]
        cluster_ranks = rank_clusters(cluster_cnts.copy(), cluster_ctrs.copy())
        return {'lbls': cluster_lbls, 'ranks': cluster_ranks, 'counts': cluster_cnts}

    decoder_utils.cluster_and_rank = deterministic_cluster_and_rank


def seed_everything(seed):
    """Seed every RNG the test pipeline can reach."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Required to make CUBLAS deterministic for matmul-heavy modules
    # (PGP aggregator's MultiheadAttention hits this path on L4/Ampere/Lovelace).
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Imports that depend on PGP_ego sys.path. Done after the path insert.
# ---------------------------------------------------------------------------
from datasets.nuScenes.nuScenes_ego_graphs_doscenes import (  # noqa: E402
    NuScenesEgoGraphsDoScenes, HISTORY_LEN, FUTURE_LEN, MIN_SCENE_SAMPLES,
)
from train_eval.initialization import initialize_prediction_model  # noqa: E402
import train_eval.utils as u  # noqa: E402

from pyquaternion import Quaternion  # noqa: E402
from nuscenes.eval.common.utils import quaternion_yaw  # noqa: E402


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
    return (np.array(ep['translation'][:2], dtype=np.float64),
            float(quaternion_yaw(Quaternion(ep['rotation']))))


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


def stage_inference(cfg, helper, test_dir, checkpoint, text_lookup, text_dim,
                    batch_size=4, num_workers=0, num_samples=1000):
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
            batch_tokens = anchor_tokens[cursor:cursor + data['ground_truth']['traj'].shape[0]]
            embs = np.stack([text_lookup.get(t, empty) for t in batch_tokens], axis=0)
            data['inputs']['text_embedding'] = torch.as_tensor(
                embs, dtype=torch.float32, device=device,
            )

            preds = model(data['inputs'])
            traj = preds['traj'].detach().cpu().numpy()
            probs = preds['probs'].detach().cpu().numpy()
            gt = data['ground_truth']['traj'].detach().cpu().numpy()
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
            if (bi + 1) % 10 == 0 or cursor == len(ds):
                print(f'  infer batch {bi+1}/{len(dl)}  ({cursor}/{len(ds)})')
    return out


def stage_submission(out, scene_tuples, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    header = ['sample_token']
    for i in range(1, FUTURE_LEN + 1):
        header += [f'x{i}', f'y{i}']
    rows, missing = [], []
    for anchor_tok, scene_tok in scene_tuples:
        if anchor_tok not in out:
            missing.append((anchor_tok, scene_tok)); continue
        chal = pgp_to_challenge_frame(out[anchor_tok]['top_traj_pgp'])
        row = [scene_tok]
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
        sc = nusc.get('scene', sc_t)
        lg = nusc.get('log', sc['log_token'])
        return lg['location']

    per_scene = []
    for anchor_tok, scene_tok in scene_tuples:
        if anchor_tok not in out:
            continue
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
    p.add_argument('--seed',            type=int, required=True,
                   help='Master seed for the whole pipeline.')
    p.add_argument('--batch_size',      type=int, default=4)
    p.add_argument('--num_workers',     type=int, default=0)
    p.add_argument('--num_samples',     type=int, default=1000)
    args = p.parse_args()

    # Seed everything BEFORE we touch any model or kmeans path.
    seed_everything(args.seed)
    patch_kmeans_determinism(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg['version'] = 'v1.0-test'

    helper = make_test_helper(args.test_root)
    nusc = helper.data
    print(f'[main] seed={args.seed}')
    print(f'[main] {len(nusc.scene)} test scenes loaded')
    scene_tuples = scenes_used_for_eval(nusc)
    print(f'[main] {len(scene_tuples)} scenes meet >= {MIN_SCENE_SAMPLES} samples')

    text_lookup, text_dim = build_text_lookup(helper, scene_tuples, args.text_emb_pkl)

    # Re-seed right before inference so any helper init that consumed RNG above
    # doesn't shift the model's draw sequence between seeds.
    seed_everything(args.seed)

    out = stage_inference(
        cfg, helper, args.test_preproc, args.checkpoint,
        text_lookup, text_dim,
        batch_size=args.batch_size, num_workers=args.num_workers,
        num_samples=args.num_samples,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'inference_cache.pkl'), 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    stage_submission(out, scene_tuples, os.path.join(args.out_dir, 'submission.csv'))
    agg, _ = stage_self_eval(
        out, scene_tuples, helper, args.test_root,
        args.doscenes_repo, os.path.join(args.out_dir, 'self_eval_metrics.json'),
    )

    with open(os.path.join(args.out_dir, 'seed.json'), 'w') as f:
        json.dump({'seed': args.seed,
                   'num_samples': args.num_samples,
                   'batch_size': args.batch_size,
                   'checkpoint': args.checkpoint}, f, indent=2)

    print('\n========== Test split results (deterministic text-conditioned) ==========')
    print(f'   seed               = {args.seed}')
    print(f'   N scenes evaluated = {len([t for t in scene_tuples if t[0] in out])}')
    for k in ['ade_2s', 'ade_4s', 'ade_6s', 'fde', 'miss_rate',
              'speed_error', 'ahe', 'fhe', 'offroad', 'offroad_rate', 'offyaw']:
        print(f'   {k:14s} = {agg[k]:.6f}')
    print('=========================================================================')


if __name__ == '__main__':
    main()
