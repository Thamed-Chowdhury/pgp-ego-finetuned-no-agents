"""
Merge two DSVT results_nusc.json files into one (union of sample tokens).
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True,
                    help='List of results_nusc.json paths to merge.')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    merged = {'meta': None, 'results': {}}
    for p in args.inputs:
        with open(p) as f:
            d = json.load(f)
        if merged['meta'] is None:
            merged['meta'] = d.get('meta', {})
        for tok, dets in d['results'].items():
            if tok not in merged['results']:
                merged['results'][tok] = dets
        print(f'  {p}: {len(d["results"])} samples; cum={len(merged["results"])}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(merged, f)
    print(f'Wrote merged results: {args.out} ({len(merged["results"])} samples)')


if __name__ == '__main__':
    main()
