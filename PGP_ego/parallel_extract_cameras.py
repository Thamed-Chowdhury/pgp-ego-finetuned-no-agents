"""
Parallel CAM_FRONT image extraction from nuScenes trainval camera archives.

Identifies all CAM_FRONT image paths needed for the annotated val scenes,
then extracts them from the relevant tgz archives using multiprocessing
(one worker per archive). Each worker streams through its archive and
extracts only the matching files.

Typical runtime: 5-15 minutes with all 10 workers running in parallel.

Usage:
  cd /teamspace/studios/this_studio
  python3 -u PGP_ego/parallel_extract_cameras.py [--high_ade_only] [--dry_run]
"""

import argparse, os, sys, tarfile, time
from pathlib import Path
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, 'PGP_ego')

from nuscenes.nuscenes import NuScenes

DATA_ROOT  = 'nuscenes_data'
ANNOT_CSV  = 'annotated_doscenes.csv'
N_FRAMES   = 4

HIGH_ADE = [44, 297, 298, 285, 165, 56, 67, 45, 68, 211, 220,
            292, 58, 42, 284, 27, 172, 124, 154]


def get_needed_filenames(nusc, target_scenes: list) -> set:
    """Collect ALL CAM_FRONT image paths for every keyframe in each target scene.
    We extract every frame (not just last N) because the 'worst window' sample
    used by the pipeline can be anywhere within the scene."""
    sc_name_to_token = {s['name']: s['token'] for s in nusc.scene}
    needed = set()
    for sc_num in target_scenes:
        sc_name = f"scene-{sc_num:04d}"
        sc_tok = sc_name_to_token.get(sc_name)
        if not sc_tok:
            continue
        scene = nusc.get('scene', sc_tok)
        # Walk the full sample chain from first to last
        sample_tok = scene['first_sample_token']
        while sample_tok:
            sample = nusc.get('sample', sample_tok)
            sd = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            fname = sd['filename']
            dest = os.path.join(DATA_ROOT, fname)
            if not os.path.exists(dest):
                needed.add(fname)
            sample_tok = sample['next']
    return needed


def extract_from_tgz(args):
    """Worker: scan one tgz archive and extract any member in needed_set."""
    tgz_idx, tgz_path, needed_set, dry_run = args
    tag = f"[trainval{tgz_idx:02d}]"
    if not os.path.exists(tgz_path):
        print(f"{tag} File not found, skipping.", flush=True)
        return set()

    found = set()
    try:
        with tarfile.open(tgz_path, 'r:gz') as tar:
            for member in tar:
                if member.name not in needed_set:
                    continue
                dest = os.path.join(DATA_ROOT, member.name)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if dry_run:
                    print(f"{tag} [dry] {member.name}", flush=True)
                else:
                    tar.extract(member, path=DATA_ROOT, filter='data')
                    print(f"{tag} ✓ {member.name}", flush=True)
                found.add(member.name)
    except Exception as e:
        print(f"{tag} ERROR: {e}", flush=True)

    if found:
        print(f"{tag} Done — extracted {len(found)} files.", flush=True)
    else:
        print(f"{tag} Done — no needed files found.", flush=True)
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--high_ade_only', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--workers', type=int, default=10,
                        help='Number of parallel archive workers (default 10)')
    args = parser.parse_args()

    print("Loading nuScenes metadata ...", flush=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=DATA_ROOT, verbose=False)

    df = pd.read_csv(ANNOT_CSV).dropna(subset=['Instruction'])
    df['scene_num'] = df['Scene Number'].astype(float).astype(int)

    ann_scenes = sorted(set(df['scene_num'].unique()))
    if args.high_ade_only:
        target = [s for s in HIGH_ADE if s in set(ann_scenes)]
    else:
        target = ann_scenes

    print(f"Target scenes: {len(target)}", flush=True)
    needed = get_needed_filenames(nusc, target)

    if not needed:
        print("All needed images already exist. Nothing to do.", flush=True)
        return

    print(f"Need to extract {len(needed)} images from archives.", flush=True)
    for p in sorted(needed):
        print(f"  {p}", flush=True)

    archive_args = [
        (i, os.path.join(DATA_ROOT, f"trainval{i:02d}_camera.tgz"), needed, args.dry_run)
        for i in range(1, 11)
    ]

    print(f"\nStarting {args.workers} parallel workers ...", flush=True)
    t0 = time.time()
    with Pool(processes=args.workers) as pool:
        results = pool.map(extract_from_tgz, archive_args)

    all_found = set()
    for r in results:
        all_found |= r

    elapsed = time.time() - t0
    missing = needed - all_found
    print(f"\n{'='*60}", flush=True)
    print(f"Extracted {len(all_found)}/{len(needed)} images in {elapsed:.0f}s", flush=True)
    if missing:
        print(f"WARNING: {len(missing)} files not found:", flush=True)
        for p in sorted(missing):
            print(f"  {p}", flush=True)
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
