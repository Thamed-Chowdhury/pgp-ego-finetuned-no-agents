"""
Non-ego variant of inject_lidar_agents_trainval.py.

Injects DSVT detections into the non-ego PGP training pickles (named
"<instance>_<sample>.pickle"). For each pickle, the anchor is the sample
token in the filename, and we walk back 4 prev samples for history.
"""

import argparse
import os
import pickle
import shutil
import sys
import json

from nuscenes.nuscenes import NuScenes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from inject_lidar_agents import build_scene_rows, pack_into_pickle  # noqa: E402


def anchor_history(nusc, anchor_token, n_history=4):
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
    ap.add_argument('--results_nusc', required=True)
    ap.add_argument('--in_dir', required=True,
                    help='Source dir of <instance>_<sample>.pickle files (e.g. pgp_preprocessed).')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--trainval_root', default='/teamspace/studios/this_studio/nuscenes_data')
    ap.add_argument('--map_extent', nargs=4, type=float, default=[-50, 50, -20, 80])
    args = ap.parse_args()

    print(f'[inject-nonego] Loading nuScenes v1.0-trainval ...')
    ns = NuScenes('v1.0-trainval', dataroot=args.trainval_root, verbose=False)

    print(f'[inject-nonego] Loading DSVT detections from {args.results_nusc} ...')
    with open(args.results_nusc) as f:
        results = json.load(f)['results']
    print(f'  detections in: {len(results)} samples')

    os.makedirs(args.out_dir, exist_ok=True)

    # Copy everything that isn't a per-window pickle (stats, etc.)
    for fn in os.listdir(args.in_dir):
        src = os.path.join(args.in_dir, fn)
        if not os.path.isfile(src):
            continue
        # window pickles look like "<32hex>_<32hex>.pickle"
        if fn.endswith('.pickle') and '_' in fn and not fn.startswith('ego_'):
            continue
        dst = os.path.join(args.out_dir, fn)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)

    pickles = sorted(f for f in os.listdir(args.in_dir)
                     if f.endswith('.pickle') and '_' in f and not f.startswith('ego_'))
    print(f'[inject-nonego] {len(pickles)} non-ego window pickles to patch')

    n_patched = n_skipped = 0
    n_veh = n_ped = 0
    n_no_dets = 0
    for i, fn in enumerate(pickles):
        # filename "<instance>_<sample>.pickle"
        stem = fn[:-len('.pickle')]
        try:
            _instance, sample = stem.split('_', 1)
        except ValueError:
            n_skipped += 1
            continue

        try:
            chain = anchor_history(ns, sample, n_history=4)
        except Exception as e:
            print(f'  skip {sample}: history error {e}')
            n_skipped += 1
            continue
        if len(chain) < 5:
            chain = [chain[0]] * (5 - len(chain)) + chain

        if not any(t in results for t in chain):
            n_no_dets += 1

        scene_rec = {'sample_tokens': chain, 'scene_name': sample}
        veh_rows, ped_rows = build_scene_rows(scene_rec, results, ns, args.map_extent)

        src = os.path.join(args.in_dir, fn)
        dst = os.path.join(args.out_dir, fn)
        with open(src, 'rb') as f:
            d = pickle.load(f)
        pack_into_pickle(d, veh_rows, ped_rows)
        with open(dst, 'wb') as f:
            pickle.dump(d, f)

        n_patched += 1
        n_veh += len(veh_rows)
        n_ped += len(ped_rows)
        if (i + 1) % 1000 == 0 or (i + 1) == len(pickles):
            print(f'  [{i+1}/{len(pickles)}] patched={n_patched} skipped={n_skipped}'
                  f' veh_tracks={n_veh} ped_tracks={n_ped} no_dets_anchors={n_no_dets}')

    print(f'\n[inject-nonego] Done.')
    print(f'  patched={n_patched} skipped={n_skipped}')
    print(f'  total vehicle tracks: {n_veh}')
    print(f'  total pedestrian tracks: {n_ped}')
    print(f'  anchors with zero detections across 5 frames: {n_no_dets}')
    print(f'  out_dir: {args.out_dir}')


if __name__ == '__main__':
    main()
