"""
Like preprocess.py but skips compute_stats (we copy stats.pickle from the
ego preprocessed dir instead) and only runs the extract phase.
"""

import argparse
import os
import shutil

import yaml
from train_eval.preprocessor import preprocess_data


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', required=True)
parser.add_argument('-r', '--data_root', required=True)
parser.add_argument('-d', '--data_dir', required=True)
parser.add_argument('--copy_stats_from', default='/teamspace/studios/this_studio/pgp_ego_preprocessed/stats.pickle')
args = parser.parse_args()

os.makedirs(args.data_dir, exist_ok=True)
stats_dst = os.path.join(args.data_dir, 'stats.pickle')
if not os.path.exists(stats_dst):
    print(f'[extract-only] Copying stats {args.copy_stats_from} -> {stats_dst}')
    shutil.copyfile(args.copy_stats_from, stats_dst)
else:
    print(f'[extract-only] stats.pickle already present at {stats_dst}')

with open(args.config) as f:
    cfg = yaml.safe_load(f)

preprocess_data(cfg, args.data_root, args.data_dir,
                compute_stats=False, extract=True)
