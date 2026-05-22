"""
Visualize PGP ego-vehicle predictions for specific nuScenes scenes.

Scenes are identified by nuScenes scene number (e.g., --scenes 28 31 61
maps to scene-0028, scene-0031, scene-0061). The prediction frame is placed
4 s (8 keyframes) from the start of each scene.

3-panel output per scene:
  1. HD map (lane graph from preprocessed features)
  2. K=10 sampled predicted ego trajectories
  3. Ground-truth ego trajectory

ADE / FDE metrics are printed and displayed in the figure title.
"""

import argparse
import os
import sys
import yaml
import pickle

# Disable Ray's memory monitor so its workers aren't killed when the node is memory-tight
# (nuScenes metadata alone occupies ~13 GB; the Ray worker itself uses <50 MB)
os.environ.setdefault('RAY_memory_monitor_refresh_ms', '0')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from nuscenes import NuScenes

sys.path.insert(0, os.path.dirname(__file__))
from train_eval.initialization import initialize_prediction_model, initialize_dataset, get_specific_args
import train_eval.utils as u

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAP_EXTENT = [-50, 50, -20, 80]   # [x_left, x_right, y_behind, y_ahead] in ego frame


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def load_model(cfg, checkpoint_path):
    model = initialize_prediction_model(
        cfg['encoder_type'], cfg['aggregator_type'], cfg['decoder_type'],
        cfg['encoder_args'], cfg['aggregator_args'], cfg['decoder_args'],
    ).float().to(device)
    model.eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


def load_ego_dataset(cfg, data_root, data_dir):
    """Loads the ego val dataset (train_val split) in load_data mode."""
    ds_type = cfg['dataset'] + '_' + cfg['agent_setting'] + '_' + cfg['input_representation']
    spec_args = get_specific_args(cfg['dataset'], data_root, cfg['version'])
    ds = initialize_dataset(ds_type, ['load_data', data_dir, cfg['val_set_args']] + spec_args)
    return ds


# ---------------------------------------------------------------------------
# Scene resolution
# ---------------------------------------------------------------------------

def get_sample_4s_by_scene_number(scene_num, nusc):
    """
    Looks up scene-{scene_num:04d} and returns the sample at 8 keyframes
    (4 s @ 2 Hz) from the start of the scene.

    Returns (sample_token, scene_token, scene_name).
    """
    scene_name = f'scene-{scene_num:04d}'
    scene = next((s for s in nusc.scene if s['name'] == scene_name), None)
    if scene is None:
        raise ValueError(f"'{scene_name}' not found in the nuScenes dataset.")

    samples = []
    s = scene['first_sample_token']
    while s:
        samples.append(s)
        s = nusc.get('sample', s)['next']

    target_frame = 8  # 4 s × 2 Hz
    if len(samples) <= target_frame + 12:
        raise ValueError(
            f"'{scene_name}' only has {len(samples)} samples; "
            f"need at least {target_frame + 13} for a full prediction window."
        )

    return samples[target_frame], scene['token'], scene_name


# ---------------------------------------------------------------------------
# Data computation
# ---------------------------------------------------------------------------

def ensure_compute_stats(ds):
    """Loads stats needed for on-the-fly feature computation (not set in load_data mode)."""
    if hasattr(ds, 'max_nodes') and hasattr(ds, 'max_nbr_nodes'):
        return
    stats = ds.load_stats()
    if not hasattr(ds, 'max_nodes'):
        ds.max_nodes = stats['num_lane_nodes']
    if not hasattr(ds, 'max_vehicles'):
        ds.max_vehicles = stats['num_vehicles']
    if not hasattr(ds, 'max_pedestrians'):
        ds.max_pedestrians = stats['num_pedestrians']
    if not hasattr(ds, 'max_nbr_nodes'):
        ds.max_nbr_nodes = stats['max_nbr_nodes']


def get_ego_data(ego_token, ds, data_dir):
    """
    Returns ego data dict for ego_token.
    Loads from pre-processed pickle if available; otherwise computes on-the-fly.
    """
    pkl_path = os.path.join(data_dir, ego_token + '.pickle')
    if os.path.isfile(pkl_path):
        print(f"  Loading from pickle: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

    ensure_compute_stats(ds)

    print(f"  Computing on-the-fly for {ego_token}")
    orig_token_list = ds.token_list
    ds.token_list = [ego_token]
    try:
        inputs = ds.get_inputs(0)
        ground_truth = ds.get_ground_truth(0)
        node_seq_gt, evf_gt = ds.get_visited_edges(0, inputs['map_representation'])
        init_node = ds.get_initial_node(inputs['map_representation'])
        inputs['init_node'] = init_node
        inputs['node_seq_gt'] = node_seq_gt
        ground_truth['evf_gt'] = evf_gt
    finally:
        ds.token_list = orig_token_list

    return {'inputs': inputs, 'ground_truth': ground_truth}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ade_fde(trajs_pred, traj_gt):
    """
    trajs_pred: [K, T, 2]  (numpy)
    traj_gt:    [T, 2]     (numpy)

    Returns dict with minADE_1/5/10, minFDE_1/5/10 computed over all K samples.
    """
    T = traj_gt.shape[0]
    # Per-sample ADE and FDE
    dists = np.linalg.norm(trajs_pred - traj_gt[None], axis=-1)   # [K, T]
    ade_per_k = dists.mean(axis=1)    # [K]
    fde_per_k = dists[:, -1]          # [K]

    sorted_ade = np.sort(ade_per_k)
    sorted_fde = np.sort(fde_per_k)

    K = len(ade_per_k)
    metrics = {}
    for k in (1, 5, 10):
        if k <= K:
            metrics[f'minADE_{k}'] = float(sorted_ade[:k].min())
            metrics[f'minFDE_{k}'] = float(sorted_fde[:k].min())

    return metrics


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_lane_nodes(ax, lane_node_feats, lane_node_masks):
    """
    Draw lane polylines. Handles batched [B, N, P, F] or unbatched [N, P, F].
    Feature layout: [x, y, yaw, is_stop_line, is_ped_crossing, is_boundary]
    """
    if torch.is_tensor(lane_node_feats):
        lnf = lane_node_feats.detach().cpu().numpy()
        lnm = lane_node_masks.detach().cpu().numpy()
    else:
        lnf = np.asarray(lane_node_feats)
        lnm = np.asarray(lane_node_masks)

    if lnf.ndim == 4:
        lnf, lnm = lnf[0], lnm[0]

    for i in range(lnf.shape[0]):
        valid = lnm[i, :, 0] == 0
        if valid.sum() < 2:
            continue
        xy = lnf[i, valid, :2]
        is_ped  = lnf[i, valid, 4].max() > 0.5
        is_stop = lnf[i, valid, 3].max() > 0.5

        if is_ped:
            color, lw, alpha = '#5599ff', 2.0, 0.75
        elif is_stop:
            color, lw, alpha = '#ff6644', 1.5, 0.70
        else:
            color, lw, alpha = '#aaaaaa', 0.9, 0.55

        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, alpha=alpha,
                solid_capstyle='round')

    ax.set_facecolor('#141414')
    ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
    ax.set_aspect('equal')


def visualize_ego_scene(scene_num, scene_name, data, model, output_dir):
    """Runs model inference, computes ADE/FDE, and saves a 3-panel PNG."""
    # Reduce num_samples to avoid OOM during Ray-based clustering (K=1000 is too large for vis)
    orig_agg_samples = model.aggregator.num_samples
    orig_dec_samples = model.decoder.num_samples
    model.aggregator.num_samples = 100
    model.decoder.num_samples = 100

    data_t = u.send_to_device(u.convert_double_to_float(u.convert2tensors(data)))

    with torch.no_grad():
        predictions = model(data_t['inputs'])

    # Restore original num_samples
    model.aggregator.num_samples = orig_agg_samples
    model.decoder.num_samples = orig_dec_samples

    trajs_pred = predictions['traj'][0].detach().cpu().numpy()   # [K, T, 2]
    traj_gt = data_t['ground_truth']['traj'][0].detach().cpu().numpy()  # [T, 2]

    # Metrics over all K samples
    metrics = compute_ade_fde(trajs_pred, traj_gt)
    metric_str = '  |  '.join(
        f"{k}: {v:.2f} m" for k, v in metrics.items()
    )
    print(f"  Metrics — {metric_str}")

    # Select 10 trajectories to display: the best-5 by ADE + 5 random others
    K_total = trajs_pred.shape[0]
    dists = np.linalg.norm(trajs_pred - traj_gt[None], axis=-1).mean(axis=1)
    best5_idx = np.argsort(dists)[:5]
    other_idx = np.setdiff1d(np.arange(K_total), best5_idx)
    random5_idx = np.random.choice(other_idx, size=min(5, len(other_idx)), replace=False)
    vis_idx = np.concatenate([best5_idx, random5_idx])
    trajs_vis = trajs_pred[vis_idx]

    lnf = data['inputs']['map_representation']['lane_node_feats']
    lnm = data['inputs']['map_representation']['lane_node_masks']
    K_vis = len(trajs_vis)
    colors = cm.plasma(np.linspace(0.1, 0.9, K_vis))

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.patch.set_facecolor('#0e0e0e')

    ade_line = '  |  '.join(
        f"{k}: {v:.2f} m"
        for k, v in metrics.items()
        if 'ADE' in k
    )
    fde_line = '  |  '.join(
        f"{k}: {v:.2f} m"
        for k, v in metrics.items()
        if 'FDE' in k
    )
    fig.suptitle(
        f'Ego  |  {scene_name}  |  t = 4 s from scene start\n'
        f'ADE → {ade_line}\n'
        f'FDE → {fde_line}',
        color='white', fontsize=11, fontweight='bold', y=1.0,
    )

    panel_titles = [
        'HD Map  (lane graph)',
        f'Predicted Trajectories  (showing {K_vis} of {K_total})',
        'Ground-Truth Trajectory',
    ]

    for ax_idx, ax in enumerate(axes):
        draw_lane_nodes(ax, lnf, lnm)
        ax.set_title(panel_titles[ax_idx], color='white', fontsize=10, pad=5)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
        ax.tick_params(colors='#666666', labelsize=7)

        ax.scatter(0, 0, s=160, color='cyan', marker='*', zorder=12)
        ax.annotate('', xy=(0, 6), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='cyan', lw=2.0))

        if ax_idx == 1:
            for k, traj in enumerate(trajs_vis):
                ax.plot(traj[:, 0], traj[:, 1], lw=1.8, color=colors[k],
                        alpha=0.55, zorder=6)
                ax.scatter(traj[-1, 0], traj[-1, 1], s=22, color=colors[k],
                           alpha=0.85, zorder=7)

        elif ax_idx == 2:
            ax.plot(traj_gt[:, 0], traj_gt[:, 1], lw=3, color='#00ff88', zorder=8)
            ax.scatter(traj_gt[-1, 0], traj_gt[-1, 1], s=90, color='#00ff88',
                       marker='D', zorder=9)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'ego_{scene_name}.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path, metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize ego-vehicle PGP predictions for nuScenes scenes by number."
    )
    parser.add_argument("-c", "--config",     required=True, help="Training config YAML")
    parser.add_argument("-r", "--data_root",  required=True, help="nuScenes data root")
    parser.add_argument("-d", "--data_dir",   required=True, help="Pre-processed data directory")
    parser.add_argument("-w", "--checkpoint", required=True, help="Ego model checkpoint (.tar)")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory for PNGs")
    parser.add_argument("-s", "--scenes", nargs='+', type=int, required=True,
                        help="nuScenes scene numbers (e.g. 28 31 61 → scene-0028 scene-0031 scene-0061)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("Loading nuScenes...")
    nusc = NuScenes(version=cfg['version'], dataroot=args.data_root, verbose=False)

    print("Loading ego model...")
    model = load_model(cfg, args.checkpoint)

    print("Loading ego dataset...")
    ds = load_ego_dataset(cfg, args.data_root, args.data_dir)

    all_metrics = {}
    for scene_num in args.scenes:
        print(f"\n--- scene-{scene_num:04d} ---")
        sample_token, scene_token, scene_name = get_sample_4s_by_scene_number(
            scene_num, nusc
        )
        ego_token = f'ego_{sample_token}'
        print(f"  sample @ 4s: {sample_token[:8]}...")

        data = get_ego_data(ego_token, ds, args.data_dir)
        path, metrics = visualize_ego_scene(
            scene_num, scene_name, data, model, args.output_dir
        )
        all_metrics[scene_name] = metrics

    print("\n" + "="*60)
    print("Summary of ADE / FDE metrics (metres):")
    print("="*60)
    for sname, m in all_metrics.items():
        print(f"\n  {sname}:")
        for k, v in m.items():
            print(f"    {k:12s} = {v:.3f} m")


if __name__ == '__main__':
    main()
