"""
Rewrite a doScenes submission.csv to match the official format described at
https://mi3-lab.github.io/doScenes_challenge :

    sample_token, instruction, x1, y1, x2, y2, ..., x12, y12

Existing submissions in this repo were written without the `instruction`
column. This script reads them and writes a sibling file with the column
inserted between `sample_token` and `x1`.

For runs where the pipeline consumed language input (the with-language
submissions of the text-conditioned model), each row's instruction is the
exact doScenes text the pipeline fed to the model (pooled across all 12
annotator CSVs with priority order d > s > sd > ds > other; empty for the
23 test scenes that had no annotation).

For runs that did not consume language input (every other submission in
this codebase), the instruction column is empty for every row.
"""

import argparse
import csv
import os
import sys
from glob import glob

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import nuscenes.eval.prediction.splits as _ns_pred_splits
_ns_pred_splits.get_prediction_challenge_split = lambda split, dataroot=None: []

from nuscenes import NuScenes


PRIORITY = {'d': 0, 's': 1, 'sd': 2, 'ds': 3}


def load_doscenes_annotations(ann_dir):
    by_scene = {}
    for fp in sorted(glob(os.path.join(ann_dir, '*.csv'))):
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
                    by_scene[scene_num] = (pr, instr)
    return {sn: ins for sn, (_, ins) in by_scene.items()}


def build_token_to_instruction(test_root, ann_dir):
    """The submission CSV in this codebase is keyed by *scene_token* (one row per
    scene). Build a scene_token -> instruction lookup. We also map the anchor
    sample_token to the same instruction in case future submissions key by
    sample_token instead."""
    nusc = NuScenes(version='v1.0-test', dataroot=test_root, verbose=False)
    anno = load_doscenes_annotations(ann_dir)
    HISTORY_LEN = 4
    MIN_SCENE_SAMPLES = 17
    out = {}
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
        instr = anno.get(scene_num, '')
        out[s['token']] = instr   # by scene_token (codebase convention)
        out[anchor] = instr       # also by anchor sample_token as a fallback
    return out


def rewrite_submission(in_csv, out_csv, token_to_instr, used_language: bool):
    with open(in_csv, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    if header[0] != 'sample_token':
        raise ValueError(f'unexpected first column in {in_csv}: {header[0]}')
    # New header: sample_token, instruction, x1, y1, ..., x12, y12
    new_header = [header[0], 'instruction'] + header[1:]
    new_rows = []
    n_with = n_without = 0
    for row in rows:
        st = row[0]
        instr = token_to_instr.get(st, '') if used_language else ''
        if instr:
            n_with += 1
        else:
            n_without += 1
        new_rows.append([st, instr] + row[1:])
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(new_header); w.writerows(new_rows)
    print(f'  {in_csv} -> {out_csv}')
    print(f'    rows={len(new_rows)}  with-instruction={n_with}  without-instruction={n_without}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test_root',  default='nuscenes_data/v1-test')
    p.add_argument('--ann_dir',    default='pgp-ego-finetuned/doScenes_repo/Annotations')
    p.add_argument('--in_csv',     required=True, help='Path to existing submission.csv')
    p.add_argument('--out_csv',    required=True, help='Output path for the fixed submission.csv')
    p.add_argument('--used_language', action='store_true',
                   help='If set, populate the instruction column with the doScenes instruction. '
                        'Otherwise leave it empty for every row.')
    args = p.parse_args()

    print('Loading test split + doScenes annotations ...')
    token_to_instr = build_token_to_instruction(args.test_root, args.ann_dir)
    print(f'  {len(token_to_instr)} anchor tokens mapped; '
          f'{sum(1 for v in token_to_instr.values() if v)} have an instruction')

    rewrite_submission(args.in_csv, args.out_csv, token_to_instr, args.used_language)


if __name__ == '__main__':
    main()
