"""
Evaluate the trained text-conditioned confidence head on the train_val split.

Reports two regimes:
  - per-pair  : every (sample, instruction) pair scored independently.
  - per-sample: one instruction per sample (random or 'first'); also reports
                averaged-over-instructions ADE/FDE for robustness.

Compared against PGP's K-means-rank baseline ('top-confidence') and the
oracle min-ADE.

Usage:
  python eval_confidence_head.py \
    --pgp_cache  ./pgp_ego_cache/train_val.pkl \
    --emb_cache  ./pgp_ego_cache/instruction_embeddings.pkl \
    --doscenes_csv .../entire_doscenes.csv \
    --head_ckpt  ./pgp_ego_output2/checkpoints/confidence_head.pt
"""

import argparse
import os
import pickle
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doscenes_data import load_doscenes
from models.text_confidence_head import TextConfidenceHead


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ade_fde(traj_k: np.ndarray, gt: np.ndarray):
    """traj_k: (T, 2)  gt: (T, 2)"""
    ade = float(np.linalg.norm(traj_k - gt, axis=-1).mean())
    fde = float(np.linalg.norm(traj_k[-1] - gt[-1]))
    return ade, fde


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pgp_cache",    required=True)
    p.add_argument("--emb_cache",    required=True)
    p.add_argument("--doscenes_csv", required=True)
    p.add_argument("--head_ckpt",    required=True)
    p.add_argument("--seed",         type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    doscenes = load_doscenes(args.doscenes_csv)
    with open(args.emb_cache, 'rb') as f:
        emb_cache = pickle.load(f)
    embeddings = emb_cache['embeddings']
    with open(args.pgp_cache, 'rb') as f:
        recs = pickle.load(f)

    ckpt = torch.load(args.head_ckpt, map_location=device, weights_only=False)
    cfg = ckpt['config']
    head = TextConfidenceHead(traj_len=cfg['traj_len'], traj_dim=cfg['traj_dim'],
                              text_dim=cfg['text_dim']).to(device)
    head.load_state_dict(ckpt['head_state_dict'])
    head.eval()

    print(f"Records: {len(recs)} ego windows on val")
    print(f"Head epoch: {ckpt.get('epoch')} | best val top-ade at train: "
          f"{ckpt.get('val_top_ade'):.4f}")

    # ---- per-pair metrics ----
    pair_head_ade, pair_head_fde = [], []
    pair_base_ade, pair_base_fde = [], []
    pair_oracle_ade = []
    # ---- per-sample (one instruction per sample) ----
    samp_head_ade_first, samp_head_fde_first = [], []
    samp_head_ade_rand,  samp_head_fde_rand  = [], []
    samp_head_ade_mean,  samp_head_fde_mean  = [], []
    # baseline doesn't depend on instruction
    samp_base_ade, samp_base_fde = [], []
    samp_oracle = []

    skipped = 0
    for rec in recs:
        instrs = doscenes.get(rec['scene_number'])
        if not instrs:
            skipped += 1
            continue
        traj = rec['traj']  # (K, T, 2)
        gt   = rec['gt']    # (T, 2)
        probs = rec['probs']

        # baseline (instruction-independent)
        base_idx = int(probs.argmax())
        b_ade, b_fde = ade_fde(traj[base_idx], gt)
        samp_base_ade.append(b_ade); samp_base_fde.append(b_fde)

        # oracle
        all_ade = np.linalg.norm(traj - gt[None], axis=-1).mean(axis=1)
        samp_oracle.append(float(all_ade.min()))

        # head over each instruction
        ades_per_instr, fdes_per_instr = [], []
        with torch.no_grad():
            traj_t = torch.from_numpy(traj).unsqueeze(0).to(device)
            for ins in instrs:
                emb = embeddings.get(ins)
                if emb is None:
                    continue
                emb_t = torch.from_numpy(emb).unsqueeze(0).to(device)
                logits = head(traj_t, emb_t).squeeze(0).cpu().numpy()
                idx = int(logits.argmax())
                a, f = ade_fde(traj[idx], gt)
                ades_per_instr.append(a); fdes_per_instr.append(f)

                pair_head_ade.append(a); pair_head_fde.append(f)
                pair_base_ade.append(b_ade); pair_base_fde.append(b_fde)
                pair_oracle_ade.append(float(all_ade.min()))

        if not ades_per_instr:
            continue
        samp_head_ade_first.append(ades_per_instr[0])
        samp_head_fde_first.append(fdes_per_instr[0])
        rand_idx = random.randrange(len(ades_per_instr))
        samp_head_ade_rand.append(ades_per_instr[rand_idx])
        samp_head_fde_rand.append(fdes_per_instr[rand_idx])
        samp_head_ade_mean.append(float(np.mean(ades_per_instr)))
        samp_head_fde_mean.append(float(np.mean(fdes_per_instr)))

    print(f"Skipped (no doScenes coverage): {skipped}")

    print("\n" + "=" * 68)
    print("PER-PAIR (every (sample, instruction) treated independently)")
    print("=" * 68)
    print(f"  pairs                  : {len(pair_head_ade)}")
    print(f"  head top-conf ADE      : {np.mean(pair_head_ade):.4f} m")
    print(f"  head top-conf FDE      : {np.mean(pair_head_fde):.4f} m")
    print(f"  baseline (K-means) ADE : {np.mean(pair_base_ade):.4f} m")
    print(f"  baseline (K-means) FDE : {np.mean(pair_base_fde):.4f} m")
    print(f"  oracle min-ADE         : {np.mean(pair_oracle_ade):.4f} m")

    print("\n" + "=" * 68)
    print("PER-SAMPLE (one instruction per sample)")
    print("=" * 68)
    print(f"  samples                : {len(samp_head_ade_first)}")
    print(f"  head ADE (1st instr)   : {np.mean(samp_head_ade_first):.4f} m")
    print(f"  head ADE (random)      : {np.mean(samp_head_ade_rand):.4f} m")
    print(f"  head ADE (mean across) : {np.mean(samp_head_ade_mean):.4f} m")
    print(f"  baseline (K-means) ADE : {np.mean(samp_base_ade):.4f} m")
    print(f"  oracle min-ADE         : {np.mean(samp_oracle):.4f} m")
    print(f"  head FDE (1st instr)   : {np.mean(samp_head_fde_first):.4f} m")
    print(f"  baseline (K-means) FDE : {np.mean(samp_base_fde):.4f} m")
    print("=" * 68)


if __name__ == '__main__':
    main()
