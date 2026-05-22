"""
Inject VLM-predicted surrounding agents into the doScenes test pickles.

For each scene in the VLM predictions file:
  - For each detected agent, compile its 5-frame (forward, right) positions.
  - Transform each frame's position from the per-frame ego coordinate system
    to the ANCHOR's PGP local frame (+y forward, +x right) using global ego
    poses from the test split's ego_pose table.
  - Discard agents outside the model's map extent.
  - Compute motion states: v (m/s), a (m/s^2), yaw_rate (rad/s, derived from
    direction-of-motion heading change).
  - Write each agent as a 5-timestep x 5-feature row into vehicles/pedestrians,
    update the corresponding masks, and save the patched pickle.

Output pickles go to --out_dir (a copy of --in_dir with vehicles/pedestrians
overwritten for the scenes we have VLM predictions for).
"""

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion


def correct_yaw(yaw):
    if yaw < 0:
        yaw += 2 * np.pi
    return yaw


def get_ego_pose_for_sample(ns, s_t):
    sample = ns.get('sample', s_t)
    sd = ns.get('sample_data', sample['data']['LIDAR_TOP'])
    ep = ns.get('ego_pose', sd['ego_pose_token'])
    x, y = ep['translation'][:2]
    yaw = quaternion_yaw(Quaternion(ep['rotation']))
    yaw = correct_yaw(yaw)
    return float(x), float(y), float(yaw)


def vlm_to_global(forward, right, ex, ey, eyaw):
    """(forward, right) in ego frame at (ex, ey, eyaw) -> (gx, gy)."""
    gx = ex + forward * np.cos(eyaw) + right * np.sin(eyaw)
    gy = ey + forward * np.sin(eyaw) - right * np.cos(eyaw)
    return gx, gy


def global_to_anchor_local(gx, gy, anchor):
    """Match PGP's NuScenesVector.global_to_local convention (+y forward, +x right)."""
    ox, oy, oyaw = anchor
    dx = gx - ox
    dy = gy - oy
    c, s = np.cos(np.pi / 2 - oyaw), np.sin(np.pi / 2 - oyaw)
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    return float(lx), float(ly)


def build_agent_rows(vlm_record, ns, map_extent, t_h=2):
    """
    Returns (vehicle_rows, pedestrian_rows). Each row: ndarray shape (5, 5)
    [x, y, v, a, yaw_rate] in the PGP anchor-local frame. Missing timesteps
    are zero-filled — masks are computed downstream.
    """
    sample_tokens = vlm_record['sample_tokens']
    assert len(sample_tokens) == 2 * t_h + 1, f"expected {2*t_h+1} samples"

    ego_poses = [get_ego_pose_for_sample(ns, st) for st in sample_tokens]
    anchor_pose = ego_poses[-1]  # samples[4] = anchor

    # Group VLM detections by (id, class)
    by_agent = {}
    for fi, fr in enumerate(vlm_record['vlm_result']['frames']):
        for a in fr['agents']:
            key = (int(a['id']), 'pedestrian' if a['class'].startswith('ped') else 'vehicle')
            gx, gy = vlm_to_global(a['forward_m'], a['right_m'], *ego_poses[fi])
            lx, ly = global_to_anchor_local(gx, gy, anchor_pose)
            by_agent.setdefault(key, [None] * 5)[fi] = (lx, ly)

    vehicles, peds = [], []
    for (aid, cls), seq in by_agent.items():
        # Need a position at the most recent timestep (anchor) — otherwise PGP's
        # discard_poses_outside_extent will likely drop or misuse the row.
        # We keep agents even if missing some past frames; the mask will hide them.
        if all(p is None for p in seq):
            continue

        # Build 5x5 array
        row = np.zeros((5, 5), dtype=np.float64)
        valid = [p is not None for p in seq]
        positions = []
        for fi, p in enumerate(seq):
            if p is not None:
                row[fi, 0] = p[0]
                row[fi, 1] = p[1]
                positions.append((fi, p))

        # Discard if no position is inside the model's map extent
        in_extent = False
        for fi, (lx, ly) in positions:
            if map_extent[0] <= lx <= map_extent[1] and map_extent[2] <= ly <= map_extent[3]:
                in_extent = True
                break
        if not in_extent:
            continue

        # Motion states from successive positions
        dt = 0.5  # 2 Hz keyframes
        v_series = np.zeros(5)
        heading_series = np.zeros(5)
        for fi in range(1, 5):
            if valid[fi] and valid[fi - 1]:
                dx = row[fi, 0] - row[fi - 1, 0]
                dy = row[fi, 1] - row[fi - 1, 1]
                d = float(np.hypot(dx, dy))
                v_series[fi] = d / dt
                if d > 1e-3:
                    heading_series[fi] = np.arctan2(dy, dx)
        a_series = np.zeros(5)
        for fi in range(1, 5):
            if valid[fi] and valid[fi - 1]:
                a_series[fi] = (v_series[fi] - v_series[fi - 1]) / dt
        yaw_rate_series = np.zeros(5)
        for fi in range(1, 5):
            if valid[fi] and valid[fi - 1]:
                dh = heading_series[fi] - heading_series[fi - 1]
                # wrap to [-pi, pi]
                dh = (dh + np.pi) % (2 * np.pi) - np.pi
                yaw_rate_series[fi] = dh / dt

        row[:, 2] = v_series
        row[:, 3] = a_series
        row[:, 4] = yaw_rate_series

        # Zero out missing timesteps (per-row masking handled below)
        for fi in range(5):
            if not valid[fi]:
                row[fi, :] = 0.0

        (peds if cls == 'pedestrian' else vehicles).append((row, valid))

    return vehicles, peds


def pack_into_pickle(d, vehicles, peds):
    sar = d['inputs']['surrounding_agent_representation']
    veh_shape = sar['vehicles'].shape       # (max_vehicles, 5, 5)
    ped_shape = sar['pedestrians'].shape    # (max_pedestrians, 5, 5)

    new_veh = np.zeros(veh_shape, dtype=sar['vehicles'].dtype)
    new_veh_mask = np.ones(veh_shape, dtype=sar['vehicle_masks'].dtype)
    new_ped = np.zeros(ped_shape, dtype=sar['pedestrians'].dtype)
    new_ped_mask = np.ones(ped_shape, dtype=sar['pedestrian_masks'].dtype)

    for i, (row, valid) in enumerate(vehicles[:veh_shape[0]]):
        new_veh[i] = row
        for t, v in enumerate(valid):
            if v:
                new_veh_mask[i, t, :] = 0.0
    for i, (row, valid) in enumerate(peds[:ped_shape[0]]):
        new_ped[i] = row
        for t, v in enumerate(valid):
            if v:
                new_ped_mask[i, t, :] = 0.0

    sar['vehicles'] = new_veh
    sar['vehicle_masks'] = new_veh_mask
    sar['pedestrians'] = new_ped
    sar['pedestrian_masks'] = new_ped_mask

    if 'agent_node_masks' in d['inputs']:
        d['inputs']['agent_node_masks'] = _recompute_agent_node_masks(d)


def _recompute_agent_node_masks(d, dist_thresh: float = 10.0):
    """Mirror NuScenesGraphs.get_agent_node_masks. Returns
    {'vehicles': (N, V), 'pedestrians': (N, P)} with 0 if the last agent pose
    is within dist_thresh of any valid node pose, 1 otherwise.
    """
    inputs = d['inputs']
    lane_node_feats = inputs['map_representation']['lane_node_feats']
    lane_node_masks = inputs['map_representation']['lane_node_masks']
    sar = inputs['surrounding_agent_representation']
    vehicles = sar['vehicles']
    veh_masks = sar['vehicle_masks']
    peds = sar['pedestrians']
    ped_masks = sar['pedestrian_masks']

    N = lane_node_feats.shape[0]
    V = vehicles.shape[0]
    P = peds.shape[0]
    veh_node_masks = np.ones((N, V), dtype=lane_node_feats.dtype)
    ped_node_masks = np.ones((N, P), dtype=lane_node_feats.dtype)

    for i in range(N):
        node_valid = (lane_node_masks[i, :, 0] == 0)
        if not node_valid.any():
            continue
        node_locs = lane_node_feats[i, node_valid, :2]

        for j in range(V):
            if (veh_masks[j] == 0).any():
                vloc = vehicles[j, -1, :2]
                d = np.min(np.linalg.norm(node_locs - vloc, axis=1))
                if d <= dist_thresh:
                    veh_node_masks[i, j] = 0.0
        for j in range(P):
            if (ped_masks[j] == 0).any():
                ploc = peds[j, -1, :2]
                d = np.min(np.linalg.norm(node_locs - ploc, axis=1))
                if d <= dist_thresh:
                    ped_node_masks[i, j] = 0.0
    return {'vehicles': veh_node_masks, 'pedestrians': ped_node_masks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--in_dir', required=True, help='Original test pickle dir')
    ap.add_argument('--out_dir', required=True, help='Destination dir for patched pickles')
    ap.add_argument('--test_root', required=True, help='v1-test root for ego_pose lookup')
    ap.add_argument('--map_extent', nargs=4, type=float, default=[-50, 50, -20, 80])
    args = ap.parse_args()

    print(f"Loading NuScenes test ...")
    ns = NuScenes('v1.0-test', dataroot=args.test_root, verbose=False)

    os.makedirs(args.out_dir, exist_ok=True)
    # Copy stats.pickle and any non-anchor files unchanged
    for fn in os.listdir(args.in_dir):
        src = os.path.join(args.in_dir, fn)
        dst = os.path.join(args.out_dir, fn)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)

    with open(args.predictions) as f:
        recs = [json.loads(l) for l in f if l.strip()]

    n_patched = 0
    for r in recs:
        if not r.get('vlm_result'):
            continue
        anchor = r['anchor_token']
        pkl_path = os.path.join(args.out_dir, f'ego_{anchor}.pickle')
        if not os.path.exists(pkl_path):
            print(f'  [miss] no pickle for anchor {anchor}')
            continue
        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)
        vehicles, peds = build_agent_rows(r, ns, args.map_extent)
        pack_into_pickle(d, vehicles, peds)
        with open(pkl_path, 'wb') as f:
            pickle.dump(d, f)
        n_patched += 1
        v_in = len(vehicles)
        p_in = len(peds)
        print(f"  patched {r['scene_name']}: {v_in} vehicles, {p_in} pedestrians in-extent")

    print(f"\nPatched {n_patched} scene pickles in {args.out_dir}")


if __name__ == '__main__':
    main()
