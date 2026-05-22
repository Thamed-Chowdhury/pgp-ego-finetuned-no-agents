"""
Inject CenterPoint LiDAR-detected agents into the doScenes test pickles.

Input:
  - results_nusc.json from OpenPCDet test.py (CenterPoint-PointPillar).
    Standard nuScenes detection submission format. Box centers are in the
    GLOBAL frame.
  - Test split full150_scenes.json (the same one used by the VLM pipeline).

For each scene:
  1. For each of the 5 keyframes (anchor + 4 history), grab detected objects.
  2. Track agents across the 5 frames via nearest-neighbor in the ANCHOR's
     local frame, using each detection's predicted velocity for gating.
  3. Build per-agent 5-step (x, y, v, a, yaw_rate) rows in the anchor PGP
     local frame, write into vehicles / pedestrians tensors, write masks.
  4. Recompute agent_node_masks consistent with NuScenesGraphs.

Output: a copy of pgp_ego_test_preprocessed/ with patched pickles.
"""

import argparse
import json
import os
import pickle
import shutil
from typing import Dict, List, Tuple

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion


# Classes from nuScenes detection task → PGP {vehicle, pedestrian} buckets.
VEHICLE_CLASSES = {
    'car', 'truck', 'bus', 'trailer',
    'motorcycle', 'bicycle', 'construction_vehicle',
}
PEDESTRIAN_CLASSES = {'pedestrian'}

# Detection score floor (CenterPoint outputs are well-calibrated).
MIN_SCORE = 0.3


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
    return float(x), float(y), float(correct_yaw(yaw))


def global_to_anchor_local(gx, gy, anchor):
    """Match PGP NuScenesVector.global_to_local: +y forward, +x right."""
    ox, oy, oyaw = anchor
    dx = gx - ox
    dy = gy - oy
    c, s = np.cos(np.pi / 2 - oyaw), np.sin(np.pi / 2 - oyaw)
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    return float(lx), float(ly)


def cls_bucket(name: str):
    if name in VEHICLE_CLASSES:
        return 'vehicle'
    if name in PEDESTRIAN_CLASSES:
        return 'pedestrian'
    return None


def build_scene_rows(scene_rec, results_by_sample, ns, map_extent):
    """
    Returns (vehicle_rows, pedestrian_rows). Each row: 5x5 ndarray
    [x, y, v, a, yaw_rate] in the PGP anchor-local frame plus a boolean
    valid-mask per timestep.
    """
    sample_tokens = scene_rec['sample_tokens']  # 5 samples: history (0..3) + anchor (4)
    ego_poses = [get_ego_pose_for_sample(ns, st) for st in sample_tokens]
    anchor_pose = ego_poses[-1]

    # Per-frame list of detections in anchor-local frame, with class + velocity.
    # frames[t] = list of dicts {x, y, cls, vlocal_xy, score}
    frames: List[List[Dict]] = []
    for fi, st in enumerate(sample_tokens):
        ex, ey, eyaw = ego_poses[fi]
        det_list = []
        for det in results_by_sample.get(st, []):
            if det['detection_score'] < MIN_SCORE:
                continue
            cls = cls_bucket(det['detection_name'])
            if cls is None:
                continue
            gx, gy = det['translation'][0], det['translation'][1]
            lx, ly = global_to_anchor_local(gx, gy, anchor_pose)
            vx, vy = det['velocity']
            # rotate global velocity into anchor local frame
            c, s = np.cos(np.pi / 2 - anchor_pose[2]), np.sin(np.pi / 2 - anchor_pose[2])
            vl_x = c * vx + s * vy
            vl_y = -s * vx + c * vy
            det_list.append({
                'x': lx, 'y': ly, 'cls': cls,
                'vx_l': vl_x, 'vy_l': vl_y,
                'score': det['detection_score'],
            })
        frames.append(det_list)

    # Greedy tracking across consecutive frames: match each det at t-1 to its
    # nearest unmatched det at t of the same class within (v*dt + 2 m) radius.
    # An agent track is a list of length 5 with None for missing frames.
    DT = 0.5
    GATE = 4.0  # base gating distance (m); inflated by predicted speed*dt

    tracks: List[List] = []
    # Initialize tracks from frame 0
    for det in frames[0]:
        tracks.append([det] + [None] * 4)

    for fi in range(1, 5):
        unmatched_dets = list(range(len(frames[fi])))
        for tr in tracks:
            last_idx = None
            for j in range(fi - 1, -1, -1):
                if tr[j] is not None:
                    last_idx = j
                    break
            if last_idx is None:
                continue
            prev = tr[last_idx]
            dt = (fi - last_idx) * DT
            # Predict using local-frame velocity
            px = prev['x'] + prev['vx_l'] * dt
            py = prev['y'] + prev['vy_l'] * dt
            speed = float(np.hypot(prev['vx_l'], prev['vy_l']))
            gate = GATE + speed * dt
            best_j, best_d = -1, gate
            for j in unmatched_dets:
                d = frames[fi][j]
                if d['cls'] != prev['cls']:
                    continue
                dist = float(np.hypot(d['x'] - px, d['y'] - py))
                if dist < best_d:
                    best_d = dist
                    best_j = j
            if best_j >= 0:
                tr[fi] = frames[fi][best_j]
                unmatched_dets.remove(best_j)

        # Any unmatched dets at frame fi seed new tracks
        for j in unmatched_dets:
            tr = [None] * 5
            tr[fi] = frames[fi][j]
            tracks.append(tr)

    # Convert tracks to 5x5 rows
    veh_rows, ped_rows = [], []
    for tr in tracks:
        if all(p is None for p in tr):
            continue
        # Determine class from first non-None
        cls = next(p['cls'] for p in tr if p is not None)

        row = np.zeros((5, 5), dtype=np.float64)
        valid = [p is not None for p in tr]

        # Position columns
        for fi, p in enumerate(tr):
            if p is not None:
                row[fi, 0] = p['x']
                row[fi, 1] = p['y']

        # Discard if no pose inside map extent
        in_extent = False
        for fi, p in enumerate(tr):
            if p is None:
                continue
            if map_extent[0] <= p['x'] <= map_extent[1] and \
               map_extent[2] <= p['y'] <= map_extent[3]:
                in_extent = True
                break
        if not in_extent:
            continue

        # Motion states derived from consecutive valid positions
        v_series = np.zeros(5)
        head_series = np.zeros(5)
        for fi in range(1, 5):
            if valid[fi] and valid[fi - 1]:
                dx = row[fi, 0] - row[fi - 1, 0]
                dy = row[fi, 1] - row[fi - 1, 1]
                d = float(np.hypot(dx, dy))
                v_series[fi] = d / DT
                if d > 1e-3:
                    head_series[fi] = float(np.arctan2(dy, dx))
        a_series = np.zeros(5)
        for fi in range(1, 5):
            if valid[fi] and valid[fi - 1]:
                a_series[fi] = (v_series[fi] - v_series[fi - 1]) / DT
        yawrate_series = np.zeros(5)
        for fi in range(1, 5):
            if valid[fi] and valid[fi - 1]:
                dh = (head_series[fi] - head_series[fi - 1] + np.pi) % (2 * np.pi) - np.pi
                yawrate_series[fi] = dh / DT

        row[:, 2] = v_series
        row[:, 3] = a_series
        row[:, 4] = yawrate_series
        for fi in range(5):
            if not valid[fi]:
                row[fi, :] = 0.0

        (ped_rows if cls == 'pedestrian' else veh_rows).append((row, valid))

    return veh_rows, ped_rows


def pack_into_pickle(d, vehicles, peds):
    sar = d['inputs']['surrounding_agent_representation']
    veh_shape = sar['vehicles'].shape
    ped_shape = sar['pedestrians'].shape

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
    inputs = d['inputs']
    lane_node_feats = inputs['map_representation']['lane_node_feats']
    lane_node_masks = inputs['map_representation']['lane_node_masks']
    sar = inputs['surrounding_agent_representation']
    vehicles = sar['vehicles']; veh_masks = sar['vehicle_masks']
    peds = sar['pedestrians']; ped_masks = sar['pedestrian_masks']

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
                d_ = np.min(np.linalg.norm(node_locs - vloc, axis=1))
                if d_ <= dist_thresh:
                    veh_node_masks[i, j] = 0.0
        for j in range(P):
            if (ped_masks[j] == 0).any():
                ploc = peds[j, -1, :2]
                d_ = np.min(np.linalg.norm(node_locs - ploc, axis=1))
                if d_ <= dist_thresh:
                    ped_node_masks[i, j] = 0.0
    return {'vehicles': veh_node_masks, 'pedestrians': ped_node_masks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes_json', required=True,
                    help='full150_scenes.json (sample_tokens per scene)')
    ap.add_argument('--results_nusc', required=True,
                    help='OpenPCDet results_nusc.json')
    ap.add_argument('--in_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--test_root', required=True)
    ap.add_argument('--map_extent', nargs=4, type=float,
                    default=[-50, 50, -20, 80])
    args = ap.parse_args()

    print('Loading nuScenes test ...')
    ns = NuScenes('v1.0-test', dataroot=args.test_root, verbose=False)

    print('Loading CenterPoint results ...')
    with open(args.results_nusc) as f:
        results = json.load(f)['results']
    print(f'  detections in: {len(results)} samples')

    print('Loading scenes list ...')
    with open(args.scenes_json) as f:
        scenes = json.load(f)
    print(f'  scenes: {len(scenes)}')

    # Copy unchanged
    os.makedirs(args.out_dir, exist_ok=True)
    for fn in os.listdir(args.in_dir):
        src = os.path.join(args.in_dir, fn)
        dst = os.path.join(args.out_dir, fn)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)

    n_patched = n_veh_total = n_ped_total = 0
    for sc in scenes:
        pkl_path = os.path.join(args.out_dir, f"ego_{sc['anchor_token']}.pickle")
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)
        vehs, peds = build_scene_rows(sc, results, ns, args.map_extent)
        pack_into_pickle(d, vehs, peds)
        with open(pkl_path, 'wb') as f:
            pickle.dump(d, f)
        n_patched += 1
        n_veh_total += len(vehs); n_ped_total += len(peds)
        print(f"  {sc['scene_name']}: {len(vehs)} veh, {len(peds)} ped in-extent")

    print(f"\nPatched {n_patched} pickles | total veh tracks: {n_veh_total} | "
          f"total ped tracks: {n_ped_total}  (out dir: {args.out_dir})")


if __name__ == '__main__':
    main()
