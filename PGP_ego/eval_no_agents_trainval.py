"""
Evaluate the no-agents PGP checkpoint on the nuScenes trainval VAL split
(cfg['test_set_args'], split='val') with surrounding agents zeroed out at
inference (matching training-time behaviour).

Reports top-confidence ADE/FDE plus best-of-K minADE/minFDE for reference,
mirroring the Table 3 layout in the technical note.

Usage:
    python eval_no_agents_trainval.py \
        -c configs/pgp_gatx2_lvm_no_agents.yml \
        -r /teamspace/studios/this_studio/nuscenes_data \
        -d /teamspace/studios/this_studio/pgp_preprocessed \
        -w /teamspace/studios/this_studio/pgp_output_no_agents_ranked/checkpoints/best.tar
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
from no_agents_utils import zero_agents_in_inputs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(cfg, checkpoint_path, num_samples=100):
    model = initialize_prediction_model(
        cfg['encoder_type'], cfg['aggregator_type'], cfg['decoder_type'],
        cfg['encoder_args'], cfg['aggregator_args'], cfg['decoder_args'],
    ).float().to(device)
    model.eval()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.aggregator.num_samples = num_samples
    model.decoder.num_samples = num_samples
    print(f"[load] checkpoint: {checkpoint_path}")
    print(f"[load] num_samples = {num_samples}")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("-r", "--data_root", required=True)
    p.add_argument("-d", "--data_dir", required=True)
    p.add_argument("-w", "--checkpoint", required=True)
    p.add_argument("--split", default="val",
                   help="Which dataset split to use: 'val' (default; trainval val) or 'train_val' or 'train'.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--out_json", default=None,
                   help="Optional path to write aggregate metrics as JSON.")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"[main] loading model from {args.checkpoint} ...")
    model = load_model(cfg, args.checkpoint, num_samples=args.num_samples)

    ds_type = cfg['dataset'] + '_' + cfg['agent_setting'] + '_' + cfg['input_representation']
    spec_args = get_specific_args(cfg['dataset'], args.data_root, cfg['version'])

    # Choose the dataset args block. 'val' → cfg['test_set_args'] (trainval val).
    if args.split == 'val':
        ds_args = dict(cfg['test_set_args'])
    elif args.split == 'train_val':
        ds_args = dict(cfg['val_set_args'])
    elif args.split == 'train':
        ds_args = dict(cfg['train_set_args'])
    else:
        raise ValueError(f"Unknown split: {args.split}")
    ds_args['random_flips'] = False

    print(f"[main] loading split={args.split} from {args.data_dir} ...")
    ds = initialize_dataset(ds_type, ['load_data', args.data_dir, ds_args] + spec_args)
    print(f"[main] {len(ds)} samples")

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    top_ades_full, top_ades_2, top_ades_4, top_ades_6 = [], [], [], []
    top_fdes = []
    min_ades = []
    min_fdes = []

    print(f"\n[eval] running on {device} (agents zeroed)")
    processed = 0
    with torch.no_grad():
        for bi, data in enumerate(dl):
            data = u.send_to_device(u.convert_double_to_float(data))
            data['inputs'] = zero_agents_in_inputs(data['inputs'])
            preds = model(data['inputs'])

            traj_pred = preds['traj'].detach().cpu().numpy()  # (B,K,T,2)
            probs = preds['probs'].detach().cpu().numpy()     # (B,K)
            traj_gt = data['ground_truth']['traj'].detach().cpu().numpy()  # (B,T,2)

            B, K, T, _ = traj_pred.shape
            for b in range(B):
                pred = traj_pred[b]
                prob = probs[b]
                gt = traj_gt[b]

                top_k = int(np.argmax(prob))
                top_traj = pred[top_k]
                step_err = np.linalg.norm(top_traj - gt, axis=-1)  # (T,)

                top_ades_full.append(float(step_err.mean()))
                # nuScenes is 2 Hz → 4 steps = 2s, 8 = 4s, 12 = 6s.
                top_ades_2.append(float(step_err[:4].mean()))
                top_ades_4.append(float(step_err[:8].mean()))
                top_ades_6.append(float(step_err[:12].mean()))
                top_fdes.append(float(step_err[-1]))

                dists = np.linalg.norm(pred - gt[None], axis=-1).mean(axis=1)
                min_ades.append(float(dists.min()))
                min_fdes.append(float(np.linalg.norm(pred[:, -1, :] - gt[-1], axis=-1).min()))

            processed += B
            if (bi + 1) % 20 == 0:
                print(f"  batch {bi+1:>4}  processed={processed:>6}  "
                      f"top-ADE={np.mean(top_ades_full):.4f}  "
                      f"minADE={np.mean(min_ades):.4f}")

    n = len(top_ades_full)
    results = {
        'n_samples': n,
        'top_conf_ade_2s': float(np.mean(top_ades_2)),
        'top_conf_ade_4s': float(np.mean(top_ades_4)),
        'top_conf_ade_6s': float(np.mean(top_ades_6)),
        'top_conf_fde_6s': float(np.mean(top_fdes)),
        'min_ade_k10':     float(np.mean(min_ades)),
        'min_fde_k10':     float(np.mean(min_fdes)),
        'top_conf_ade_full_mean': float(np.mean(top_ades_full)),
        'top_conf_ade_full_std':  float(np.std(top_ades_full)),
    }

    print("\n" + "=" * 70)
    print(f" no-agents PGP, trainval {args.split} split, agents zeroed at inference")
    print("=" * 70)
    print(f"  N samples         : {n}")
    print(f"  Top-conf ADE @ 2s : {results['top_conf_ade_2s']:.4f} m")
    print(f"  Top-conf ADE @ 4s : {results['top_conf_ade_4s']:.4f} m")
    print(f"  Top-conf ADE @ 6s : {results['top_conf_ade_6s']:.4f} m")
    print(f"  Top-conf FDE @ 6s : {results['top_conf_fde_6s']:.4f} m")
    print(f"  minADE  (best/10) : {results['min_ade_k10']:.4f} m  (reference)")
    print(f"  minFDE  (best/10) : {results['min_fde_k10']:.4f} m  (reference)")
    print("=" * 70)

    if args.out_json:
        import json
        os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[main] wrote {args.out_json}")


if __name__ == '__main__':
    main()
