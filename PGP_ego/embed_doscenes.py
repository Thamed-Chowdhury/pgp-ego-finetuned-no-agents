"""
Pre-compute and cache sentence embeddings for every doScenes instruction.

Output: <cache_dir>/instruction_embeddings.pkl with
  {'model': str, 'dim': int, 'embeddings': {instruction_text: np.ndarray}}

Run once; both the trainer and the eval script load this cache.
"""

import argparse
import os
import pickle

import numpy as np

from doscenes_data import load_doscenes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="D:/DriveX_PGP/doScenes-VLM-Planning-main/data/doScenes/entire_doscenes.csv")
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = p.parse_args()

    print(f"Loading doScenes from {args.csv}")
    doscenes = load_doscenes(args.csv)
    instructions = sorted({i for instrs in doscenes.values() for i in instrs})
    print(f"  {len(doscenes)} scenes, {len(instructions)} unique instructions")

    print(f"Loading text encoder: {args.model}")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(args.model)
    enc.eval()

    print("Encoding ...")
    vecs = enc.encode(instructions, batch_size=64, show_progress_bar=True,
                      convert_to_numpy=True, normalize_embeddings=False)
    dim = int(vecs.shape[1])
    print(f"  dim = {dim}")

    embeddings = {instr: vec.astype(np.float32) for instr, vec in zip(instructions, vecs)}

    os.makedirs(args.cache_dir, exist_ok=True)
    out = os.path.join(args.cache_dir, 'instruction_embeddings.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'model': args.model, 'dim': dim, 'embeddings': embeddings}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {len(embeddings)} embeddings -> {out}")


if __name__ == '__main__':
    main()
