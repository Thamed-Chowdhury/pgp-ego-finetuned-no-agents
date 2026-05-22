"""
Evaluate ego-vehicle PGP predictions on the nuScenes TEST (val) split.

For every sample in the test (val) split:
  - Run model inference
  - Pick the trajectory with the highest predicted confidence (argmax probs)
  - Compute ADE / FDE vs ground truth
  - Also compute miss-rate (top-1, threshold 2m) and best-of-K reference metrics

No doScenes / LLM / VLM reranking — straight PGP-ego top-confidence selection.

Usage:
  python eval_top_confidence_test.py \
    -c configs/pgp_ego_gatx2_lvm_traversal.yml \
    -r <nuscenes_root> \
    -d <preprocessed_data_dir> \
    -w <checkpoint.tar>
"""

import argparse
import os

os.environ.setdefault('RAY_memory_monitor_refresh_ms', '0')

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from train_eval.initialization import (
    initialize_prediction_model, initialize_dataset, get_specific_args,
)
import train_eval.utils as u

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(cfg, checkpoint_path, num_samples=100):
    model = initialize_prediction_model(
        cfg['encoder_type'], cfg['aggregator_type'], cfg['decoder_type'],
        cfg['encoder_args'], cfg['aggregator_args'], cfg['decoder_args'],
    ).float().to(device)
    model.eval()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.aggregator.num_samples = num_samples
    model.decoder.num_samples = num_samples
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Reported val metric in ckpt: {ckpt.get('val_metric', 'n/a')}")
    print(f"  num_samples set to {num_samples} for both aggregator and decoder")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",     required=True)
    parser.add_argument("-r", "--data_root",  required=True)
    parser.add_argument("-d", "--data_dir",   required=True)
    parser.add_argument("-w", "--checkpoint", required=True)
    parser.add_argument("-o", "--output_dir", default=None,
                        help="Optional dir to write per-sample CSV + summary.")
    parser.add_argument("--batch_size",       type=int, default=32)
    parser.add_argument("--num_workers",      type=int, default=4)
    parser.add_argument("--num_samples",      type=int, default=100)
    parser.add_argument("--version", default=None)
    parser.add_argument("--miss_thresh", type=float, default=2.0,
                        help="L2 threshold (m) for miss-rate at the final timestep.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    version = args.version if args.version else cfg['version']

    print(f"Loading model from {args.checkpoint} ...")
    model = load_model(cfg, args.checkpoint, num_samples=args.num_samples)

    print(f"Loading dataset (TEST/val split) from {args.data_dir} ...")
    ds_type = cfg['dataset'] + '_' + cfg['agent_setting'] + '_' + cfg['input_representation']
    spec_args = get_specific_args(cfg['dataset'], args.data_root, version)
    test_args = dict(cfg['test_set_args'])
    test_args['random_flips'] = False
    test_ds = initialize_dataset(ds_type, ['load_data', args.data_dir, test_args] + spec_args)
    print(f"  Test set size: {len(test_ds)} samples")

    dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    top_ades, top_fdes = [], []
    min_ades5, min_fdes5 = [], []
    min_ades10, min_fdes10 = [], []
    top_misses = []     # top-1 miss (FDE > thresh)
    miss5, miss10 = [], []

    total = len(test_ds)
    processed = 0

    print(f"\nRunning evaluation on {device} ...")
    print(f"{'Batch':>6}  {'Processed':>10}  {'AvgTopADE':>10}  {'AvgTopFDE':>10}  {'AvgMinADE5':>10}")
    print("-" * 62)

    rows = []  # (sample_token, top_ade, top_fde, min_ade5, min_fde5, min_ade10, min_fde10, top_miss)

    with torch.no_grad():
        for batch_idx, data in enumerate(dl):
            data = u.send_to_device(u.convert_double_to_float(data))
            predictions = model(data['inputs'])

            traj_pred = predictions['traj'].detach().cpu().numpy()    # [B, K, T, 2]
            probs     = predictions['probs'].detach().cpu().numpy()   # [B, K]
            traj_gt   = data['ground_truth']['traj'].detach().cpu().numpy()  # [B, T, 2]

            sample_tokens = data['inputs'].get('sample_token', [None] * traj_pred.shape[0])

            B = traj_pred.shape[0]
            for b in range(B):
                pred  = traj_pred[b]   # [K, T, 2]
                prob  = probs[b]
                gt    = traj_gt[b]     # [T, 2]

                # Top-confidence trajectory
                top_k = int(np.argmax(prob))
                top_traj = pred[top_k]
                top_ade  = float(np.linalg.norm(top_traj - gt, axis=-1).mean())
                top_fde  = float(np.linalg.norm(top_traj[-1] - gt[-1]))
                top_miss = float(top_fde > args.miss_thresh)
                top_ades.append(top_ade); top_fdes.append(top_fde); top_misses.append(top_miss)

                # min over top-K (K=5, K=10) using model's own confidence ranking
                K = pred.shape[0]
                order = np.argsort(-prob)
                for K_keep, ade_buf, fde_buf, miss_buf in (
                    (min(5,  K), min_ades5,  min_fdes5,  miss5),
                    (min(10, K), min_ades10, min_fdes10, miss10),
                ):
                    sub = pred[order[:K_keep]]
                    ades = np.linalg.norm(sub - gt[None], axis=-1).mean(axis=1)
                    fdes = np.linalg.norm(sub[:, -1, :] - gt[-1], axis=-1)
                    ade_buf.append(float(ades.min()))
                    fde_buf.append(float(fdes.min()))
                    miss_buf.append(float(fdes.min() > args.miss_thresh))

                rows.append((
                    sample_tokens[b] if sample_tokens is not None else '',
                    top_ade, top_fde, top_miss,
                    min_ades5[-1], min_fdes5[-1],
                    min_ades10[-1], min_fdes10[-1],
                ))

            processed += B
            if (batch_idx + 1) % 5 == 0 or processed == total:
                print(f"{batch_idx+1:>6}  {processed:>10}  "
                      f"{np.mean(top_ades):>10.4f}  {np.mean(top_fdes):>10.4f}  "
                      f"{np.mean(min_ades5):>10.4f}")

    n = len(top_ades)
    print("\n" + "=" * 72)
    print(f"FINAL RESULTS  (ego model, TEST/val split, top-confidence trajectory)")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 72)
    print(f"  Samples evaluated  : {n}")
    print(f"  Top-1 ADE           : {np.mean(top_ades):.4f} m  (std={np.std(top_ades):.4f})")
    print(f"  Top-1 FDE           : {np.mean(top_fdes):.4f} m  (std={np.std(top_fdes):.4f})")
    print(f"  Top-1 MissRate@{args.miss_thresh:.1f}m : {np.mean(top_misses):.4f}")
    print(f"  minADE_5  (ref)     : {np.mean(min_ades5):.4f} m")
    print(f"  minFDE_5  (ref)     : {np.mean(min_fdes5):.4f} m")
    print(f"  MissRate_5@{args.miss_thresh:.1f}m   : {np.mean(miss5):.4f}")
    print(f"  minADE_10 (ref)     : {np.mean(min_ades10):.4f} m")
    print(f"  minFDE_10 (ref)     : {np.mean(min_fdes10):.4f} m")
    print(f"  MissRate_10@{args.miss_thresh:.1f}m  : {np.mean(miss10):.4f}")
    print("=" * 72)

    print("\nTop-1 ADE percentile breakdown (m):")
    for p in [25, 50, 75, 90, 95]:
        print(f"  p{p:2d}: {np.percentile(top_ades, p):.4f}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, 'top1_per_sample.csv')
        with open(csv_path, 'w') as f:
            f.write("sample_token,top1_ade,top1_fde,top1_miss,minADE5,minFDE5,minADE10,minFDE10\n")
            for r in rows:
                f.write(",".join(str(x) for x in r) + "\n")

        summary_path = os.path.join(args.output_dir, 'top1_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Samples: {n}\n")
            f.write(f"Top-1 ADE:  {np.mean(top_ades):.6f}\n")
            f.write(f"Top-1 FDE:  {np.mean(top_fdes):.6f}\n")
            f.write(f"Top-1 MissRate@{args.miss_thresh}m: {np.mean(top_misses):.6f}\n")
            f.write(f"minADE_5:   {np.mean(min_ades5):.6f}\n")
            f.write(f"minFDE_5:   {np.mean(min_fdes5):.6f}\n")
            f.write(f"MissRate_5@{args.miss_thresh}m: {np.mean(miss5):.6f}\n")
            f.write(f"minADE_10:  {np.mean(min_ades10):.6f}\n")
            f.write(f"minFDE_10:  {np.mean(min_fdes10):.6f}\n")
            f.write(f"MissRate_10@{args.miss_thresh}m: {np.mean(miss10):.6f}\n")
        print(f"\nWrote per-sample CSV → {csv_path}")
        print(f"Wrote summary        → {summary_path}")


if __name__ == '__main__':
    main()
