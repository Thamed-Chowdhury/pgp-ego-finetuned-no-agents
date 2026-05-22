"""
Build a DSVT-style info pkl with the sample tokens needed for PGP non-ego
training that are NOT yet in the existing DSVT detections cache.
"""

import argparse
import json
import os
import pickle

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trainval_root', default='/teamspace/studios/this_studio/nuscenes_data')
    ap.add_argument('--dsvt_data_root', default='/teamspace/studios/this_studio/DSVT/data/nuscenes/v1.0-trainval')
    ap.add_argument('--existing_results', default='/teamspace/studios/this_studio/DSVT/output/dsvt_models/dsvt_plain_1f_onestage_nusences_pgp_train/pgp_train/eval/epoch_no_number/val/default/final_result/data/results_nusc.json')
    ap.add_argument('--out', default='/teamspace/studios/this_studio/DSVT/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_pgp_nonego_delta.pkl')
    args = ap.parse_args()

    print('[build-pgp-nonego] Loading nuScenes v1.0-trainval ...')
    ns = NuScenes('v1.0-trainval', dataroot=args.trainval_root, verbose=False)

    needed = set()
    for split in ['train', 'train_val']:
        toks = get_prediction_challenge_split(split, dataroot=args.trainval_root)
        samples = set(t.split('_')[1] for t in toks)
        for s in samples:
            needed.add(s)
            cur = s
            for _ in range(4):
                prev = ns.get('sample', cur).get('prev', '')
                if not prev:
                    break
                cur = prev
                needed.add(cur)
    print(f'[build-pgp-nonego] {len(needed)} unique sample tokens needed (train + train_val, anchor + 4 prev)')

    with open(args.existing_results) as f:
        have = set(json.load(f)['results'].keys())
    delta = needed - have
    print(f'[build-pgp-nonego] {len(have)} samples already detected; delta = {len(delta)} samples')

    by_token = {}
    for name in ['nuscenes_infos_10sweeps_train.pkl',
                 'nuscenes_infos_10sweeps_val_full.pkl']:
        path = os.path.join(args.dsvt_data_root, name)
        print(f'[build-pgp-nonego] Loading {name} ...')
        with open(path, 'rb') as f:
            infos = pickle.load(f)
        for inf in infos:
            tok = inf['token']
            if tok in delta and tok not in by_token:
                by_token[tok] = inf
        print(f'  cumulative coverage: {len(by_token)} / {len(delta)}')

    missing = delta - set(by_token)
    if missing:
        print(f'[build-pgp-nonego] WARNING: {len(missing)} delta tokens not in info pkls')
    else:
        print('[build-pgp-nonego] All delta tokens covered.')

    out_list = list(by_token.values())
    print(f'[build-pgp-nonego] writing {len(out_list)} infos → {args.out}')
    with open(args.out, 'wb') as f:
        pickle.dump(out_list, f)
    print('[build-pgp-nonego] Done.')


if __name__ == '__main__':
    main()
