"""
Selectively extract CAM_FRONT images for annotated val scenes.

Reads nuScenes metadata to find which image files are needed, then extracts
only those files from the relevant trainval*_camera.tgz archives.

This avoids extracting the full ~170 GB of camera data when we only need
~4 images × N scenes.

Usage:
  cd /teamspace/studios/this_studio
  python3 -u PGP_ego/extract_scene_cameras.py [--dry_run]
"""

import argparse, os, sys, json, tarfile, time
from pathlib import Path

import pandas as pd

sys.path.insert(0, 'PGP_ego')

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.prediction.splits import create_splits_scenes, NUM_IN_TRAIN_VAL

DATA_ROOT   = 'nuscenes_data'
ANNOT_CSV   = 'annotated_doscenes.csv'
TGZ_DIR     = DATA_ROOT                       # .tgz files live here
N_FRAMES    = 4                               # last 2 seconds at 2 Hz
TGZ_PATTERN = 'trainval{:02d}_camera.tgz'    # 01-10

HIGH_ADE = [44, 297, 298, 285, 165, 56, 67, 45, 68, 211, 220,
            292, 58, 42, 284, 27, 172, 124, 154]


def get_needed_cam_files(nusc, target_scenes: list) -> dict:
    """
    For each scene in target_scenes, find the last N_FRAMES CAM_FRONT sample_data
    filenames. Returns {tgz_member_path: local_dest_path}.
    """
    sc_name_to_token = {s['name']: s['token'] for s in nusc.scene}
    needed = {}

    for sc_num in target_scenes:
        sc_name = f"scene-{sc_num:04d}"
        sc_tok = sc_name_to_token.get(sc_name)
        if not sc_tok:
            continue
        # Walk samples to find all keyframes in this scene
        scene = nusc.get('scene', sc_tok)
        sample_tok = scene['last_sample_token']
        count = 0
        while sample_tok and count < N_FRAMES:
            sample = nusc.get('sample', sample_tok)
            sd = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            fname = sd['filename']               # e.g. samples/CAM_FRONT/n015-...jpg
            dest = os.path.join(DATA_ROOT, fname)
            if not os.path.exists(dest):
                needed[fname] = dest
            sample_tok = sample['prev']
            count += 1

    return needed


def find_files_in_tgz(tgz_path: str, needed_set: set, dry_run: bool) -> set:
    """Open tgz_path and extract any member whose name is in needed_set."""
    found = set()
    if not os.path.exists(tgz_path):
        return found
    print(f"  Scanning {os.path.basename(tgz_path)} ...", flush=True)
    try:
        with tarfile.open(tgz_path, 'r:gz') as tar:
            members = tar.getmembers()
            to_extract = [m for m in members if m.name in needed_set]
            if not to_extract:
                print(f"    No needed files here.", flush=True)
                return found
            print(f"    Found {len(to_extract)} files to extract.", flush=True)
            for m in to_extract:
                dest = os.path.join(DATA_ROOT, m.name)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if dry_run:
                    print(f"    [dry] {m.name}", flush=True)
                else:
                    tar.extract(m, path=DATA_ROOT, filter='data')
                    print(f"    ✓ {m.name}", flush=True)
                found.add(m.name)
    except Exception as e:
        print(f"    ERROR reading {tgz_path}: {e}", flush=True)
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', action='store_true',
                        help='Print what would be extracted without extracting')
    parser.add_argument('--high_ade_only', action='store_true',
                        help='Extract only for 19 high-ADE scenes')
    args = parser.parse_args()

    print("Loading nuScenes metadata ...", flush=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=DATA_ROOT, verbose=False)

    print("Loading annotations ...", flush=True)
    df = pd.read_csv(ANNOT_CSV).dropna(subset=['Instruction'])
    df['scene_num'] = df['Scene Number'].astype(float).astype(int)

    tv_scenes = set(create_splits_scenes()['train'][:NUM_IN_TRAIN_VAL])
    all_sc_names = {s['name'] for s in nusc.scene}
    tv_scene_nums = set()
    for name in tv_scenes:
        if name in all_sc_names:
            try:
                tv_scene_nums.add(int(name.split('-')[1]))
            except ValueError:
                pass

    ann_scenes = sorted(set(df['scene_num'].unique()) & tv_scene_nums)
    if args.high_ade_only:
        target = [s for s in HIGH_ADE if s in set(ann_scenes)]
    else:
        target = ann_scenes

    print(f"Target scenes: {len(target)}", flush=True)

    needed = get_needed_cam_files(nusc, target)
    if not needed:
        print("All needed images already exist. Nothing to extract.", flush=True)
        return

    print(f"\nNeed to extract {len(needed)} images from tgz archives.", flush=True)
    if args.dry_run:
        print("(dry run — not actually extracting)", flush=True)

    needed_set = set(needed.keys())
    total_extracted = 0

    for i in range(1, 11):
        tgz = os.path.join(TGZ_DIR, TGZ_PATTERN.format(i))
        found = find_files_in_tgz(tgz, needed_set, dry_run=args.dry_run)
        total_extracted += len(found)
        needed_set -= found
        if not needed_set:
            print("\nAll needed files found.", flush=True)
            break

    if needed_set:
        print(f"\nWARNING: {len(needed_set)} files not found in any archive:", flush=True)
        for p in sorted(needed_set)[:10]:
            print(f"  {p}", flush=True)

    print(f"\nDone. Extracted {total_extracted} images.", flush=True)


if __name__ == '__main__':
    main()
