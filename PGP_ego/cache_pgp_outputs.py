"""
Cache frozen-PGP outputs for the ego model.

For each ego window in {train, train_val}, runs the model once and stores:
  - traj:         (K, T, 2)  K=10 clustered candidate trajectories
  - probs:        (K,)       cluster-rank-derived baseline probabilities
  - gt:           (T, 2)     ground-truth future trajectory (agent frame)
  - sample_token: str        nuScenes sample token (window's t=0)
  - scene_number: int        nuScenes scene index (parsed from scene name)

Outputs:
  <cache_dir>/train.pt
  <cache_dir>/train_val.pt

LVM is stochastic, so we draw one trajectory set per window and freeze it.
This makes training reproducible and avoids re-running PGP each epoch.
"""

import argparse
import os
import sys
import pickle

os.environ.setdefault('RAY_memory_monitor_refresh_ms', '0')

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from train_eval.initialization import initialize_prediction_model, initialize_dataset, get_specific_args
import train_eval.utils as u

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(cfg, checkpoint_path, num_samples=100):
    model = initialize_prediction_model(
        cfg['encoder_type'], cfg['aggregator_type'], cfg['decoder_type'],
        cfg['encoder_args'], cfg['aggregator_args'], cfg['decoder_args'],
    ).float().to(device)
    model.eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.aggregator.num_samples = num_samples
    model.decoder.num_samples = num_samples
    return model


def split_args(cfg, split: str):
    """Pull the right *_set_args block and disable random_flips for caching."""
    key = {
        'train': 'train_set_args',
        'train_val': 'val_set_args',
        'val': 'test_set_args',
    }[split]
    sa = dict(cfg[key])
    sa['random_flips'] = False
    return sa


def cache_split(model, cfg, args, split: str, sample_token_list):
    """Run the model over a split, return list[dict] in dataset order."""
    ds_type = cfg['dataset'] + '_' + cfg['agent_setting'] + '_' + cfg['input_representation']
    spec_args = get_specific_args(cfg['dataset'], args.data_root, args.version)
    sa = split_args(cfg, split)
    ds = initialize_dataset(ds_type, ['load_data', args.data_dir, sa] + spec_args)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    # token order = dataset order (DataLoader with shuffle=False)
    tokens = [t.split('_', 1)[1] for t in ds.token_list]
    assert len(tokens) == len(ds), 'token / dataset length mismatch'

    print(f"  [{split}] dataset size: {len(ds)}")

    out = []
    cursor = 0
    with torch.no_grad():
        for bi, data in enumerate(dl):
            data = u.send_to_device(u.convert_double_to_float(data))
            preds = model(data['inputs'])
            traj  = preds['traj'].detach().cpu().numpy()    # (B, K, T, 2)
            probs = preds['probs'].detach().cpu().numpy()   # (B, K)
            gt    = data['ground_truth']['traj'].detach().cpu().numpy()  # (B, T, 2)

            B = traj.shape[0]
            for b in range(B):
                out.append({
                    'sample_token': tokens[cursor + b],
                    'traj': traj[b].astype(np.float32),
                    'probs': probs[b].astype(np.float32),
                    'gt': gt[b].astype(np.float32),
                })
            cursor += B
            if (bi + 1) % 5 == 0 or cursor == len(ds):
                print(f"    batch {bi+1:>4}  cached {cursor}/{len(ds)}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config",     required=True)
    p.add_argument("-r", "--data_root",  required=True)
    p.add_argument("-d", "--data_dir",   required=True)
    p.add_argument("-w", "--checkpoint", required=True)
    p.add_argument("-o", "--cache_dir",  required=True)
    p.add_argument("--version", default=None)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--splits", default='train,train_val')
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.version is None:
        args.version = cfg['version']

    print(f"Loading model: {args.checkpoint}")
    model = load_model(cfg, args.checkpoint, num_samples=args.num_samples)

    os.makedirs(args.cache_dir, exist_ok=True)

    # We need scene_number too — attach it after fetching all samples (needs nuScenes table)
    from nuscenes.nuscenes import NuScenes
    print(f"Loading nuScenes tables (version={args.version})")
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)

    for split in args.splits.split(','):
        split = split.strip()
        if not split:
            continue
        print(f"\n=== Caching split: {split} ===")
        records = cache_split(model, cfg, args, split, None)
        # Add scene_number per record
        for rec in records:
            sample = nusc.get('sample', rec['sample_token'])
            scene  = nusc.get('scene', sample['scene_token'])
            rec['scene_number'] = int(scene['name'].split('-')[-1])
        out_path = os.path.join(args.cache_dir, f'{split}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  wrote {len(records)} records -> {out_path}")


if __name__ == '__main__':
    main()
