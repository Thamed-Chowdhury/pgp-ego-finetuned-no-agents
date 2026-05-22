"""
Build a DSVT-style info pkl that contains only the keyframes PGP-ego training
needs (anchor + 4 prev for each of the 1832 ego pickles), drawn from the union
of nuscenes_infos_10sweeps_train.pkl and nuscenes_infos_10sweeps_val_full.pkl.

Output is a flat list of info dicts ready to drop into the
DSVT/data/nuscenes/v1.0-trainval/ tree and reference from a YAML.
"""

import argparse
import os
import pickle

from nuscenes.nuscenes import NuScenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pgp_preprocessed', default='/teamspace/studios/this_studio/pgp_ego_preprocessed')
    ap.add_argument('--trainval_root', default='/teamspace/studios/this_studio/nuscenes_data')
    ap.add_argument('--dsvt_data_root', default='/teamspace/studios/this_studio/DSVT/data/nuscenes/v1.0-trainval')
    ap.add_argument('--out', default='/teamspace/studios/this_studio/DSVT/data/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_pgp_train.pkl')
    args = ap.parse_args()

    print('[build-pgp-info] Loading nuScenes v1.0-trainval ...')
    ns = NuScenes('v1.0-trainval', dataroot=args.trainval_root, verbose=False)

    anchors = []
    for f in os.listdir(args.pgp_preprocessed):
        if f.startswith('ego_') and f.endswith('.pickle'):
            anchors.append(f[len('ego_'):-len('.pickle')])
    print(f'[build-pgp-info] {len(anchors)} anchor pickles found')

    needed = set()
    for a in anchors:
        s = a
        needed.add(s)
        for _ in range(4):
            prev = ns.get('sample', s).get('prev', '')
            if not prev:
                break
            s = prev
            needed.add(s)
    print(f'[build-pgp-info] {len(needed)} unique sample tokens needed (anchor + 4 prev)')

    by_token = {}
    for name in ['nuscenes_infos_10sweeps_train.pkl',
                 'nuscenes_infos_10sweeps_val_full.pkl']:
        path = os.path.join(args.dsvt_data_root, name)
        print(f'[build-pgp-info] Loading {name} ...')
        with open(path, 'rb') as f:
            infos = pickle.load(f)
        for inf in infos:
            tok = inf['token']
            if tok in needed and tok not in by_token:
                by_token[tok] = inf
        print(f'  cumulative coverage: {len(by_token)} / {len(needed)}')

    missing = needed - set(by_token)
    if missing:
        print(f'[build-pgp-info] WARNING: {len(missing)} tokens not found in info pkls')
        for t in list(missing)[:5]:
            print(f'    {t}')
    else:
        print('[build-pgp-info] All needed tokens covered.')

    out_list = list(by_token.values())
    print(f'[build-pgp-info] writing {len(out_list)} infos -> {args.out}')
    with open(args.out, 'wb') as f:
        pickle.dump(out_list, f)
    print('[build-pgp-info] Done.')


if __name__ == '__main__':
    main()
