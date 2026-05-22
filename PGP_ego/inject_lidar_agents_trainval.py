"""
Trainval variant of inject_lidar_agents.py.

Patches PGP-ego training pickles in pgp_ego_preprocessed/ with detector-quality
agents from a DSVT (OpenPCDet) results_nusc.json. Output goes to a new dir
(e.g. pgp_ego_preprocessed_dsvt/), leaving the original GT-agent pickles
untouched.

Differences vs the test-set inject script:
  - Loads v1.0-trainval (not v1.0-test).
  - Anchors and history windows are derived from the input pickle filenames
    + the nuScenes API (no scenes_json needed).
"""

import argparse
import json
import os
import pickle
import shutil
import sys

import numpy as np
from nuscenes.nuscenes import NuScenes

# Reuse the per-scene packing / tracking helpers from the test-set script.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from inject_lidar_agents import (
    build_scene_rows,
    pack_into_pickle,
    MIN_SCORE,
)


def anchor_history(nusc, anchor_token, n_history=4):
    """Return [oldest, ..., second_oldest, prev, anchor] (length n_history + 1)."""
    chain = [anchor_token]
    s = anchor_token
    for _ in range(n_history):
        prev = nusc.get('sample', s).get('prev', '')
        if not prev:
            break
        s = prev
        chain.append(s)
    chain.reverse()
    return chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_nusc', required=True,
                    help='DSVT/OpenPCDet results_nusc.json (test-mode output)')
    ap.add_argument('--in_dir', default='/teamspace/studios/this_studio/pgp_ego_preprocessed')
    ap.add_argument('--out_dir', required=True,
                    help='Destination dir (created if missing).')
    ap.add_argument('--trainval_root', default='/teamspace/studios/this_studio/nuscenes_data')
    ap.add_argument('--map_extent', nargs=4, type=float,
                    default=[-50, 50, -20, 80])
    ap.add_argument('--presence_only', action='store_true',
                    help='Zero out velocity/accel/yaw_rate channels, keep positions only.')
    ap.add_argument('--only_anchor', default=None,
                    help='Optional single anchor sample_token (for testing).')
    args = ap.parse_args()

    print(f'[inject-trainval] Loading nuScenes v1.0-trainval @ {args.trainval_root} ...')
    ns = NuScenes('v1.0-trainval', dataroot=args.trainval_root, verbose=False)

    print(f'[inject-trainval] Loading DSVT detections from {args.results_nusc} ...')
    with open(args.results_nusc) as f:
        results = json.load(f)['results']
    print(f'  detections in: {len(results)} samples')

    os.makedirs(args.out_dir, exist_ok=True)

    # Copy stats.pickle and any non-ego files unchanged.
    for fn in os.listdir(args.in_dir):
        if fn.startswith('ego_') and fn.endswith('.pickle'):
            continue
        src = os.path.join(args.in_dir, fn)
        dst = os.path.join(args.out_dir, fn)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)

    pickles = sorted(f for f in os.listdir(args.in_dir)
                     if f.startswith('ego_') and f.endswith('.pickle'))
    if args.only_anchor:
        pickles = [f for f in pickles if args.only_anchor in f]
    print(f'[inject-trainval] {len(pickles)} ego pickles to patch')

    n_patched = n_skipped = 0
    n_veh_tot = n_ped_tot = 0
    n_missing_det = 0

    for i, fn in enumerate(pickles):
        anchor = fn[len('ego_'):-len('.pickle')]
        try:
            chain = anchor_history(ns, anchor, n_history=4)
        except Exception as e:
            print(f'  skip {anchor}: history error {e}')
            n_skipped += 1
            continue

        if len(chain) < 5:
            # zero-pad by repeating the oldest available frame; mask masking is
            # handled inside build_scene_rows / pack_into_pickle.
            chain = [chain[0]] * (5 - len(chain)) + chain

        # Skip if NO detections for the anchor (most extreme case)
        if not any(t in results for t in chain):
            n_missing_det += 1

        # Build a scenes-style record for build_scene_rows
        scene_rec = {'sample_tokens': chain, 'scene_name': anchor}

        veh_rows, ped_rows = build_scene_rows(scene_rec, results, ns, args.map_extent)

        if args.presence_only:
            # zero out velocity/accel/yaw_rate channels in every row
            for r, _v in veh_rows:
                r[:, 2:] = 0.0
            for r, _v in ped_rows:
                r[:, 2:] = 0.0

        src = os.path.join(args.in_dir, fn)
        dst = os.path.join(args.out_dir, fn)
        # Always start fresh from the input pickle (don't reuse a prior patched copy)
        with open(src, 'rb') as f:
            d = pickle.load(f)
        pack_into_pickle(d, veh_rows, ped_rows)
        with open(dst, 'wb') as f:
            pickle.dump(d, f)

        n_patched += 1
        n_veh_tot += len(veh_rows)
        n_ped_tot += len(ped_rows)

        if (i + 1) % 100 == 0 or (i + 1) == len(pickles):
            print(f'  [{i+1}/{len(pickles)}] patched={n_patched} skipped={n_skipped}'
                  f' veh_tracks_so_far={n_veh_tot} ped_tracks_so_far={n_ped_tot}'
                  f' anchors_with_no_dets={n_missing_det}')

    print(f'\n[inject-trainval] Done.')
    print(f'  patched={n_patched}  skipped={n_skipped}')
    print(f'  total vehicle tracks: {n_veh_tot}')
    print(f'  total pedestrian tracks: {n_ped_tot}')
    print(f'  anchors with zero detections across all 5 frames: {n_missing_det}')
    print(f'  out_dir: {args.out_dir}')


if __name__ == '__main__':
    main()
