"""
Evaluate ego-vehicle PGP predictions on the nuScenes TRAIN split.

For every sample in the training set:
  - Run model inference (num_samples=100, num_clusters=10)
  - Pick the trajectory with the highest predicted confidence (argmax probs)
  - Compute ADE and FDE against ground truth

The train split is the one that overlaps with doScenes driver annotations, so
this is the relevant set for downstream LLM-reranking experiments.

Reports: per-sample metrics + dataset-level averages.

Usage:
  python eval_top_confidence.py \
    -c configs/pgp_ego_gatx2_lvm_traversal.yml \
    -r <nuscenes_root> \
    -d <preprocessed_data_dir> \
    -w <checkpoint.tar>
"""

import argparse
import os
import sys

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
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  num_samples set to {num_samples} for both aggregator and decoder")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",     required=True)
    parser.add_argument("-r", "--data_root",  required=True)
    parser.add_argument("-d", "--data_dir",   required=True)
    parser.add_argument("-w", "--checkpoint", required=True)
    parser.add_argument("--batch_size",       type=int, default=32)
    parser.add_argument("--num_workers",      type=int, default=4)
    parser.add_argument("--num_samples",      type=int, default=100,
                        help="Trajectory samples for clustering (100 is sufficient for eval)")
    parser.add_argument("--version", default=None,
                        help="Override nuScenes version (folder name under data_root). "
                             "Defaults to cfg['version'].")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    version = args.version if args.version else cfg['version']

    print(f"Loading model from {args.checkpoint} ...")
    model = load_model(cfg, args.checkpoint, num_samples=args.num_samples)

    print(f"Loading dataset (TRAIN split) from {args.data_dir} ...")
    ds_type = cfg['dataset'] + '_' + cfg['agent_setting'] + '_' + cfg['input_representation']
    spec_args = get_specific_args(cfg['dataset'], args.data_root, version)
    # Use the nuScenes train split. Disable random_flips for deterministic eval.
    train_eval_args = dict(cfg['train_set_args'])
    train_eval_args['random_flips'] = False
    val_ds = initialize_dataset(ds_type, ['load_data', args.data_dir, train_eval_args] + spec_args)
    print(f"  Dataset size: {len(val_ds)} samples")

    dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    top_ades = []
    top_fdes = []
    min_ades = []  # best-of-K for reference
    min_fdes = []

    total = len(val_ds)
    processed = 0

    print(f"\nRunning evaluation on {device} ...")
    print(f"{'Batch':>6}  {'Processed':>10}  {'AvgTopADE':>10}  {'AvgMinADE':>10}")
    print("-" * 50)

    with torch.no_grad():
        for batch_idx, data in enumerate(dl):
            data = u.send_to_device(u.convert_double_to_float(data))
            predictions = model(data['inputs'])

            traj_pred = predictions['traj'].detach().cpu().numpy()   # [B, K, T, 2]
            probs     = predictions['probs'].detach().cpu().numpy()  # [B, K]
            traj_gt   = data['ground_truth']['traj'].detach().cpu().numpy()  # [B, T, 2]

            B = traj_pred.shape[0]
            for b in range(B):
                pred  = traj_pred[b]   # [K, T, 2]
                prob  = probs[b]       # [K]
                gt    = traj_gt[b]     # [T, 2]

                # Top-confidence trajectory
                top_k = int(np.argmax(prob))
                top_traj = pred[top_k]  # [T, 2]
                top_ade  = float(np.linalg.norm(top_traj - gt, axis=-1).mean())
                top_fde  = float(np.linalg.norm(top_traj[-1] - gt[-1]))
                top_ades.append(top_ade)
                top_fdes.append(top_fde)

                # Best-of-K for reference
                dists   = np.linalg.norm(pred - gt[None], axis=-1).mean(axis=1)  # [K]
                min_ade = float(dists.min())
                min_fde = float(np.linalg.norm(pred[:, -1, :] - gt[-1], axis=-1).min())
                min_ades.append(min_ade)
                min_fdes.append(min_fde)

            processed += B
            if (batch_idx + 1) % 10 == 0 or processed == total:
                avg_top_ade = np.mean(top_ades)
                avg_min_ade = np.mean(min_ades)
                print(f"{batch_idx+1:>6}  {processed:>10}  {avg_top_ade:>10.4f}  {avg_min_ade:>10.4f}")

    print("\n" + "=" * 60)
    print("FINAL RESULTS  (ego model, TRAIN split, highest-confidence traj)")
    print("=" * 60)
    print(f"  Samples evaluated : {len(top_ades)}")
    print(f"  Top-confidence ADE: {np.mean(top_ades):.4f} m  (std={np.std(top_ades):.4f})")
    print(f"  Top-confidence FDE: {np.mean(top_fdes):.4f} m  (std={np.std(top_fdes):.4f})")
    print(f"  minADE (best-of-K) : {np.mean(min_ades):.4f} m  (reference)")
    print(f"  minFDE (best-of-K) : {np.mean(min_fdes):.4f} m  (reference)")
    print("=" * 60)

    # Percentile breakdown for top-conf ADE
    print("\nTop-confidence ADE percentile breakdown:")
    for p in [25, 50, 75, 90, 95]:
        print(f"  p{p:2d}: {np.percentile(top_ades, p):.4f} m")


if __name__ == '__main__':
    main()
