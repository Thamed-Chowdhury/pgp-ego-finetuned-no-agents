"""
In-process seed sweep for deterministic LVMRankedText inference.

This is the fast version of seed_sweep.py: load NuScenes test, maps, model
and dataset ONCE, then loop seeds. Each seed re-seeds every RNG, re-runs
inference, and re-runs the metrics aggregation. Maps and PredictHelper are
cached, so each seed iteration is just the inference + ego-metric loop.

Per-seed cost on this hardware is ~60-90 seconds vs. ~6 minutes for the
subprocess sweep.

Output layout (same as seed_sweep.py):
    results/seed_sweep_inproc/seed_<S>/inference_cache.pkl
    results/seed_sweep_inproc/seed_<S>/submission.csv
    results/seed_sweep_inproc/seed_<S>/self_eval_metrics.json
    results/seed_sweep_inproc/sweep_log.json
    results/seed_sweep_inproc/sweep_summary.csv
    results/seed_sweep_inproc/best_so_far.json
"""

import argparse
import csv
import importlib.util
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PGP_EGO_DIR = os.path.join(ROOT, 'PGP_ego')
sys.path.insert(0, PGP_EGO_DIR)

import nuscenes.eval.prediction.splits as _ns_pred_splits  # noqa: E402
_ns_pred_splits.get_prediction_challenge_split = lambda split, dataroot=None: []


# ---------------------------------------------------------------------------
# Determinism helpers (copies from run_deterministic_inference.py kept close
# to where the sweep state lives).
# ---------------------------------------------------------------------------
def seed_everything(seed):
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


def patch_kmeans_determinism():
    """Patch utils.cluster_and_rank with a version that reads the seed from a
    mutable holder. We update the holder each iteration, so the same module-
    level remote function picks up the new seed without re-decoration.
    """
    import models.decoders.utils as decoder_utils
    from sklearn.cluster import KMeans
    import ray
    from scipy.spatial.distance import cdist

    seed_holder = {'seed': 0}

    @ray.remote
    def deterministic_cluster_and_rank(k, data, seed):
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
                clu_ids_idx = c
                cluster_ids = np.delete(cluster_ids, cluster_ids[cluster_ids == cluster_ids[cluster_ids_idx]])  # placeholder
            return ranks

        # The body above purposely keeps the same logic as the upstream
        # cluster_and_rank; the only changes are np.random.seed at the top and
        # the explicit random_state in KMeans. We import and reuse the upstream
        # rank_clusters by re-implementing identically for clarity.
        def rank_clusters_clean(cluster_counts, cluster_centers):
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
        cluster_ranks = rank_clusters_clean(cluster_cnts.copy(), cluster_ctrs.copy())
        return {'lbls': cluster_lbls, 'ranks': cluster_ranks, 'counts': cluster_cnts}

    # Wrap the original cluster_traj so we can inject the current seed into the
    # remote call without changing its signature for callers.
    orig_cluster_traj = decoder_utils.cluster_traj

    import torch as _torch  # local alias to dodge any shadowing.

    def deterministic_cluster_traj(k, traj):
        batch_size = traj.shape[0]
        num_samples = traj.shape[1]
        traj_len = traj.shape[2]
        data = traj[:, :, 0::3, :]
        data = data.reshape(batch_size, num_samples, -1).detach().cpu().numpy()
        seed = seed_holder['seed']
        cluster_ops = ray.get([
            deterministic_cluster_and_rank.remote(k, data_slice, seed)
            for data_slice in data
        ])
        cluster_lbls = np.stack([co['lbls'] for co in cluster_ops], axis=0)
        cluster_counts = np.stack([co['counts'] for co in cluster_ops], axis=0)
        cluster_ranks = np.stack([co['ranks'] for co in cluster_ops], axis=0)

        device = traj.device
        lbls = _torch.as_tensor(cluster_lbls, device=device).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, traj_len, 2).long()
        traj_summed = _torch.zeros(batch_size, k, traj_len, 2, device=device).scatter_add(1, lbls, traj)
        cnt_tensor = _torch.as_tensor(cluster_counts, device=device).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, traj_len, 2)
        traj_clustered = traj_summed / cnt_tensor
        scores = 1 / _torch.as_tensor(cluster_ranks, device=device)
        scores = scores / _torch.sum(scores, dim=1)[0]
        return traj_clustered, scores

    decoder_utils.cluster_traj = deterministic_cluster_traj
    return seed_holder


# ---------------------------------------------------------------------------
# Imports from PGP_ego (after sys.path insert + the K-means patch).
# ---------------------------------------------------------------------------
seed_holder = patch_kmeans_determinism()

from datasets.nuScenes.nuScenes_ego_graphs_doscenes import (  # noqa: E402
    NuScenesEgoGraphsDoScenes, HISTORY_LEN, FUTURE_LEN, MIN_SCENE_SAMPLES,
)
from train_eval.initialization import initialize_prediction_model  # noqa: E402
import train_eval.utils as u  # noqa: E402

from pyquaternion import Quaternion  # noqa: E402
from nuscenes.eval.common.utils import quaternion_yaw  # noqa: E402
from nuscenes.map_expansion.map_api import NuScenesMap  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers ported from run_deterministic_inference.py
# ---------------------------------------------------------------------------
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
    for anchor_tok, scene_tok in scene_tuples:
        scene = nusc.get('scene', scene_tok)
        try:
            scene_num = int(scene['name'].split('-')[1])
        except Exception:
            scene_num = -1
        instr = scene_to_instr.get(scene_num, '')
        if instr and instr in instr_emb:
            lookup[anchor_tok] = instr_emb[instr]
        else:
            lookup[anchor_tok] = empty_emb
    return lookup, text_dim


def load_compute_ego_metrics(doscenes_repo):
    metrics_path = os.path.join(doscenes_repo, 'metrics.py')
    spec = importlib.util.spec_from_file_location('doscenes_metrics', metrics_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.compute_ego_metrics


def evaluate_outputs(out, scene_tuples, helper, test_root, doscenes_repo, map_cache):
    compute_ego_metrics = load_compute_ego_metrics(doscenes_repo)
    nusc = helper.data
    per_scene = []
    for anchor_tok, scene_tok in scene_tuples:
        if anchor_tok not in out:
            continue
        sc = nusc.get('scene', scene_tok)
        lg = nusc.get('log', sc['log_token'])
        location = lg['location']
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
    return agg, per_scene


def write_submission(out, scene_tuples, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    header = ['sample_token']
    for i in range(1, FUTURE_LEN + 1):
        header += [f'x{i}', f'y{i}']
    rows = []
    for anchor_tok, scene_tok in scene_tuples:
        if anchor_tok not in out:
            continue
        chal = pgp_to_challenge_frame(out[anchor_tok]['top_traj_pgp'])
        row = [scene_tok]
        for x, y in chal:
            row.extend([f'{float(x):.6f}', f'{float(y):.6f}'])
        rows.append(row)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def run_inference(model, dl, anchor_tokens, text_lookup, text_dim, device):
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
    return out


def parse_seeds(seeds_arg, range_arg):
    if range_arg:
        a, b = range_arg.split(':')
        return list(range(int(a), int(b)))
    if seeds_arg:
        return [int(s.strip()) for s in seeds_arg.split(',') if s.strip()]
    raise SystemExit('must pass --seeds or --range')


def append_sweep_log(log_path, entry):
    if os.path.isfile(log_path):
        with open(log_path) as f:
            data = json.load(f)
    else:
        data = []
    data.append(entry)
    with open(log_path, 'w') as f:
        json.dump(data, f, indent=2)


def append_csv(csv_path, entry, header):
    new = not os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow([entry.get(k, '') for k in header])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', help='comma-separated list')
    p.add_argument('--range', help='start:stop (exclusive)')
    p.add_argument('--out_root', default=os.path.join(HERE, 'results', 'seed_sweep_inproc'))
    p.add_argument('--target', type=float, default=2.65)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_samples', type=int, default=1000)
    p.add_argument('--keep_going', action='store_true')
    p.add_argument('--save_caches', action='store_true',
                   help='Persist inference_cache.pkl per seed. Default is to '
                        'only keep submission.csv + self_eval_metrics.json.')
    args = p.parse_args()

    seeds = parse_seeds(args.seeds, args.range)
    print(f'[sweep-inproc] {len(seeds)} seeds, target ADE@6s <= {args.target}')

    os.makedirs(args.out_root, exist_ok=True)
    log_path = os.path.join(args.out_root, 'sweep_log.json')
    csv_path = os.path.join(args.out_root, 'sweep_summary.csv')
    header = ['seed', 'ade_2s', 'ade_4s', 'ade_6s', 'fde', 'miss_rate',
              'offroad', 'offroad_rate', 'wall_s']

    cfg_path = os.path.join(HERE, 'configs', 'pgp_ego_gatx2_lvm_ranked_text_stage3.yml')
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg['version'] = 'v1.0-test'

    test_root = os.path.join(ROOT, 'nuscenes_data', 'v1-test')
    test_preproc = os.path.join(HERE, 'data', 'test_preproc')
    text_emb_pkl = os.path.join(HERE, 'data', 'doscenes_gemini_embeddings.pkl')
    checkpoint = os.path.join(HERE, 'checkpoints', 'stage3_best.tar')
    doscenes_repo = os.path.join(ROOT, 'pgp-ego-finetuned', 'doScenes_repo')

    print('[setup] loading NuScenes test helper ...')
    helper = make_test_helper(test_root)
    nusc = helper.data
    scene_tuples = scenes_used_for_eval(nusc)
    print(f'[setup] {len(scene_tuples)} scenes meet >= {MIN_SCENE_SAMPLES} samples')

    text_lookup, text_dim = build_text_lookup(helper, scene_tuples, text_emb_pkl)

    sa = dict(cfg['test_set_args'])
    sa['random_flips'] = False
    sa['split'] = 'doscenes_test'
    sa.pop('text_emb_pkl', None)
    ds = NuScenesEgoGraphsDoScenes('load_data', test_preproc, sa, helper)
    anchor_tokens = [t.split('_', 1)[1] for t in ds.token_list]
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('[setup] loading model ...')
    model = initialize_prediction_model(
        cfg['encoder_type'], cfg['aggregator_type'], cfg['decoder_type'],
        cfg['encoder_args'], cfg['aggregator_args'], cfg['decoder_args'],
    ).float().to(device)
    model.eval()
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.aggregator.num_samples = args.num_samples
    model.decoder.num_samples = args.num_samples
    print(f'[setup] ckpt loaded, val_metric={ckpt.get("val_metric", "?")}')

    # Pre-load every map needed.
    map_cache = {}
    needed_locations = set()
    for _, scene_tok in scene_tuples:
        sc = nusc.get('scene', scene_tok)
        needed_locations.add(nusc.get('log', sc['log_token'])['location'])
    for loc in sorted(needed_locations):
        print(f'[setup] loading map: {loc}')
        map_cache[loc] = NuScenesMap(dataroot=test_root, map_name=loc)

    best = {'seed': None, 'ade_6s': float('inf')}
    if os.path.isfile(os.path.join(args.out_root, 'best_so_far.json')):
        with open(os.path.join(args.out_root, 'best_so_far.json')) as f:
            saved = json.load(f)
            if saved.get('ade_6s') is not None:
                best = saved

    for seed in seeds:
        t0 = time.time()
        seed_everything(seed)
        seed_holder['seed'] = seed

        out = run_inference(model, dl, anchor_tokens, text_lookup, text_dim, device)
        agg, _ = evaluate_outputs(out, scene_tuples, helper, test_root, doscenes_repo, map_cache)
        dt = time.time() - t0

        out_dir = os.path.join(args.out_root, f'seed_{seed}')
        os.makedirs(out_dir, exist_ok=True)
        write_submission(out, scene_tuples, os.path.join(out_dir, 'submission.csv'))
        with open(os.path.join(out_dir, 'self_eval_metrics.json'), 'w') as f:
            json.dump({'aggregate': agg, 'n_scenes': len(out)}, f, indent=2)
        with open(os.path.join(out_dir, 'seed.json'), 'w') as f:
            json.dump({'seed': seed, 'num_samples': args.num_samples,
                       'batch_size': args.batch_size,
                       'checkpoint': checkpoint}, f, indent=2)
        if args.save_caches:
            with open(os.path.join(out_dir, 'inference_cache.pkl'), 'wb') as f:
                pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

        entry = {'seed': seed, 'wall_s': round(dt, 1),
                 **{k: agg[k] for k in ['ade_2s', 'ade_4s', 'ade_6s', 'fde',
                                        'miss_rate', 'offroad', 'offroad_rate']}}
        append_sweep_log(log_path, entry)
        append_csv(csv_path, entry, header)
        print(f'[sweep] seed={seed:>4d}  ADE@6s={agg["ade_6s"]:.6f}  '
              f'ADE@2s={agg["ade_2s"]:.4f}  '
              f'FDE={agg["fde"]:.4f}  '
              f'wall={dt:.1f}s')

        if agg['ade_6s'] < best['ade_6s']:
            best = {'seed': seed, **{k: agg[k] for k in
                    ['ade_2s', 'ade_4s', 'ade_6s', 'fde', 'miss_rate',
                     'offroad', 'offroad_rate']}}
            with open(os.path.join(args.out_root, 'best_so_far.json'), 'w') as f:
                json.dump(best, f, indent=2)

        if agg['ade_6s'] <= args.target and not args.keep_going:
            print(f'\n[sweep] TARGET MET: seed={seed} ADE@6s={agg["ade_6s"]:.6f} <= {args.target}')
            break

    print('\n========== SWEEP DONE ==========')
    print(f'best_seed   = {best["seed"]}')
    print(f'best_ade_6s = {best["ade_6s"]:.6f}')


if __name__ == '__main__':
    main()
