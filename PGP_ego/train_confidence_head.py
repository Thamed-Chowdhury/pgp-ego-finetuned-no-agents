"""
Train the text-conditioned confidence head over PGP's K=10 trajectories.

PGP (encoder + aggregator + decoder) is FROZEN. We only train the small
TextConfidenceHead on cached PGP outputs + cached doScenes instruction
embeddings.

Each (sample, instruction) pair is one training example. The label is the
trajectory index whose mean displacement to ground truth is smallest
(argmin-ADE). Multiple instructions per scene act as paraphrase augmentation.

Usage:
  python train_confidence_head.py \
    --pgp_cache_dir ./pgp_ego_cache \
    --emb_cache    ./pgp_ego_cache/instruction_embeddings.pkl \
    --doscenes_csv .../entire_doscenes.csv \
    --out_dir      ./pgp_ego_output2/checkpoints \
    --epochs 30
"""

import argparse
import os
import pickle
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doscenes_data import load_doscenes
from models.text_confidence_head import TextConfidenceHead


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def argmin_ade_label(traj: np.ndarray, gt: np.ndarray) -> int:
    """traj: (K, T, 2)  gt: (T, 2)  -> argmin_k mean_t ||traj_k - gt||."""
    diff = traj - gt[None]                       # (K, T, 2)
    dists = np.linalg.norm(diff, axis=-1).mean(axis=1)  # (K,)
    return int(dists.argmin())


class PairDataset(Dataset):
    """One example per (sample, instruction). Pre-built in memory; tiny."""

    def __init__(self, pgp_records, doscenes, embeddings, mode='train'):
        self.examples = []
        miss_scene = miss_emb = 0
        for rec in pgp_records:
            instrs = doscenes.get(rec['scene_number'])
            if not instrs:
                miss_scene += 1
                continue
            label = argmin_ade_label(rec['traj'], rec['gt'])
            for ins in instrs:
                emb = embeddings.get(ins)
                if emb is None:
                    miss_emb += 1
                    continue
                self.examples.append({
                    'traj': rec['traj'],
                    'gt': rec['gt'],
                    'label': label,
                    'emb': emb,
                    'sample_token': rec['sample_token'],
                    'instruction': ins,
                    'baseline_probs': rec['probs'],
                })
        print(f"  [{mode}] built {len(self.examples)} (sample, instruction) pairs "
              f"(missed: scene={miss_scene}, emb={miss_emb})")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            'traj':  torch.from_numpy(ex['traj']),
            'gt':    torch.from_numpy(ex['gt']),
            'emb':   torch.from_numpy(ex['emb']),
            'label': torch.tensor(ex['label'], dtype=torch.long),
            'baseline_probs': torch.from_numpy(ex['baseline_probs']),
        }


def evaluate(head, loader):
    """Return (top_ade, top_fde, base_top_ade, base_top_fde, oracle_min_ade, ce_loss)."""
    head.eval()
    top_ades, top_fdes = [], []
    base_top_ades, base_top_fdes = [], []
    oracle_min_ades = []
    losses = []
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for b in loader:
            traj  = b['traj'].to(device)             # (B, K, T, 2)
            gt    = b['gt'].to(device)               # (B, T, 2)
            emb   = b['emb'].to(device)              # (B, D)
            label = b['label'].to(device)            # (B,)
            base_probs = b['baseline_probs'].to(device)  # (B, K)

            logits = head(traj, emb)                  # (B, K)
            losses.append(crit(logits, label).item())

            pred_idx = logits.argmax(dim=1)           # (B,)
            base_idx = base_probs.argmax(dim=1)

            B, K, T, _ = traj.shape
            ar = torch.arange(B, device=device)

            sel = traj[ar, pred_idx]                  # (B, T, 2)
            ade = (sel - gt).norm(dim=-1).mean(dim=1)
            fde = (sel[:, -1] - gt[:, -1]).norm(dim=-1)
            top_ades.extend(ade.cpu().tolist())
            top_fdes.extend(fde.cpu().tolist())

            sel_b = traj[ar, base_idx]
            ade_b = (sel_b - gt).norm(dim=-1).mean(dim=1)
            fde_b = (sel_b[:, -1] - gt[:, -1]).norm(dim=-1)
            base_top_ades.extend(ade_b.cpu().tolist())
            base_top_fdes.extend(fde_b.cpu().tolist())

            all_ade = (traj - gt[:, None]).norm(dim=-1).mean(dim=2)  # (B, K)
            oracle_min_ades.extend(all_ade.min(dim=1).values.cpu().tolist())

    return (float(np.mean(top_ades)), float(np.mean(top_fdes)),
            float(np.mean(base_top_ades)), float(np.mean(base_top_fdes)),
            float(np.mean(oracle_min_ades)), float(np.mean(losses)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pgp_cache_dir", required=True)
    p.add_argument("--emb_cache",     required=True)
    p.add_argument("--doscenes_csv",  required=True)
    p.add_argument("--out_dir",       required=True)
    p.add_argument("--epochs",        type=int, default=30)
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-5)
    p.add_argument("--seed",          type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"Loading doScenes from {args.doscenes_csv}")
    doscenes = load_doscenes(args.doscenes_csv)

    print(f"Loading instruction embeddings from {args.emb_cache}")
    with open(args.emb_cache, 'rb') as f:
        emb_cache = pickle.load(f)
    embeddings = emb_cache['embeddings']
    text_dim = emb_cache['dim']
    print(f"  text_dim = {text_dim}, {len(embeddings)} embeddings")

    print(f"Loading PGP cache from {args.pgp_cache_dir}")
    with open(os.path.join(args.pgp_cache_dir, 'train.pkl'), 'rb') as f:
        train_recs = pickle.load(f)
    with open(os.path.join(args.pgp_cache_dir, 'train_val.pkl'), 'rb') as f:
        val_recs = pickle.load(f)
    print(f"  train ego windows: {len(train_recs)}, val ego windows: {len(val_recs)}")

    train_ds = PairDataset(train_recs, doscenes, embeddings, mode='train')
    val_ds   = PairDataset(val_recs,   doscenes, embeddings, mode='val')

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    K, T, D = train_recs[0]['traj'].shape
    head = TextConfidenceHead(traj_len=T, traj_dim=D, text_dim=text_dim).to(device)
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"Head trainable params: {n_params:,}")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, 'confidence_head.pt')

    best_val_ade = float('inf')
    print(f"\nTraining for {args.epochs} epochs on {device}")
    print(f"{'Ep':>3} {'TrLoss':>8} {'VlLoss':>8} {'VlTopADE':>9} {'VlBaseADE':>10} {'VlOracle':>9}")
    print("-" * 60)
    for ep in range(1, args.epochs + 1):
        head.train()
        ep_loss = 0.0
        n = 0
        for b in train_dl:
            traj  = b['traj'].to(device)
            emb   = b['emb'].to(device)
            label = b['label'].to(device)
            logits = head(traj, emb)
            loss = crit(logits, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * traj.size(0)
            n += traj.size(0)
        tr_loss = ep_loss / max(n, 1)

        v_ade, v_fde, base_ade, base_fde, oracle, v_loss = evaluate(head, val_dl)
        marker = ''
        if v_ade < best_val_ade:
            best_val_ade = v_ade
            torch.save({
                'head_state_dict': head.state_dict(),
                'config': {
                    'traj_len': T, 'traj_dim': D, 'text_dim': text_dim,
                    'emb_model': emb_cache['model'],
                },
                'epoch': ep,
                'val_top_ade': v_ade, 'val_top_fde': v_fde,
                'baseline_top_ade': base_ade, 'baseline_top_fde': base_fde,
                'oracle_min_ade': oracle,
            }, ckpt_path)
            marker = '  *'

        print(f"{ep:>3} {tr_loss:>8.4f} {v_loss:>8.4f} {v_ade:>9.4f} {base_ade:>10.4f} {oracle:>9.4f}{marker}")

    print(f"\nBest val top-confidence ADE: {best_val_ade:.4f} m")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == '__main__':
    main()
