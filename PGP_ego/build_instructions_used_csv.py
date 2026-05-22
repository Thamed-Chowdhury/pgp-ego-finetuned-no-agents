"""
Produce an `instructions_used.csv` audit file for the text-conditioned
submission. One row per test scene with the exact doScenes instruction
(and its annotator + type) that the pipeline fed to the model.

Schema:
    sample_token, scene_name, scene_number, annotator, ann_type, instruction

Empty-instruction rows correspond to scenes with no doScenes annotation;
those are the 23 scenes resolved to PGP's top-confidence via the
zero-embedding fallback.
"""

import argparse
import csv
import os
import pickle
import sys
from glob import glob

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import nuscenes.eval.prediction.splits as _ns_pred_splits
_ns_pred_splits.get_prediction_challenge_split = lambda split, dataroot=None: []

from datasets.nuScenes.nuScenes_ego_graphs_doscenes import HISTORY_LEN, MIN_SCENE_SAMPLES
from nuscenes import NuScenes


PRIORITY = {'d': 0, 's': 1, 'sd': 2, 'ds': 3}


def load_doscenes_annotations(ann_dir):
    """Pool all annotator CSVs, return scene_num -> (priority, annotator, instr, ann_type)."""
    by_scene = {}
    files = sorted(glob(os.path.join(ann_dir, '*.csv')))
    for fp in files:
        annotator = os.path.basename(fp).replace('doScenesAnnotations - ', '').replace('.csv', '')
        with open(fp, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                try:
                    scene_num = int(str(row.get('Scene Number', '')).strip())
                except Exception:
                    continue
                instr = (row.get('Instruction') or '').strip()
                if not instr:
                    continue
                ann_type = (row.get('Instruction Type') or '').strip().lower().rstrip(' "')
                pr = PRIORITY.get(ann_type, 9)
                cur = by_scene.get(scene_num)
                if (cur is None) or (pr < cur[0]):
                    by_scene[scene_num] = (pr, annotator, instr, ann_type or '?')
    return by_scene


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test_root', required=True)
    p.add_argument('--ann_dir',   required=True)
    p.add_argument('--out',       required=True)
    args = p.parse_args()

    nusc = NuScenes(version='v1.0-test', dataroot=args.test_root, verbose=False)
    anno = load_doscenes_annotations(args.ann_dir)

    rows = []
    n_anno = n_empty = 0
    for s in sorted(nusc.scene, key=lambda x: x['name']):
        samples = []
        t = s['first_sample_token']
        while t:
            samples.append(t)
            t = nusc.get('sample', t)['next']
        if len(samples) < MIN_SCENE_SAMPLES:
            continue
        anchor = samples[HISTORY_LEN]
        try:
            scene_num = int(s['name'].split('-')[1])
        except Exception:
            scene_num = -1
        ann = anno.get(scene_num)
        if ann is None:
            rows.append([anchor, s['name'], scene_num, '', '', ''])
            n_empty += 1
        else:
            _, annotator, instr, ann_type = ann
            rows.append([anchor, s['name'], scene_num, annotator, ann_type, instr])
            n_anno += 1

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sample_token', 'scene_name', 'scene_number', 'annotator', 'ann_type', 'instruction'])
        w.writerows(rows)
    print(f'wrote {len(rows)} rows to {args.out}')
    print(f'  with-annotation: {n_anno}')
    print(f'  empty (zero-emb fallback at inference): {n_empty}')


if __name__ == '__main__':
    main()
