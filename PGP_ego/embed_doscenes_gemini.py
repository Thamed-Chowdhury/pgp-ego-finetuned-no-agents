"""
Cache Gemini 'gemini-embedding-001' embeddings for every doScenes instruction
(plus a placeholder zero vector for scenes with no annotation), and build a
per-scene instruction lookup.

Output (one pickle):
    <out_pkl>:
      {
        'model':                'gemini-embedding-001',
        'dim':                  768,
        'instr_embeddings':     {instruction_text: np.ndarray(768)},
        'scene_to_instruction': {scene_num (int): instruction_text or ''},
        'empty_embedding':      np.zeros(768),
      }

Annotation pooling: for each scene, pick the best annotation across all 12
annotators in priority order d -> s -> sd -> ds -> (anything else). Empty rows
are skipped. 23 of 150 test scenes (and many trainval scenes) end up with no
annotation; those map to the empty embedding (zeros) so the model sees no text
signal.
"""

import argparse
import csv
import os
import pickle
import sys
import time
from glob import glob

import numpy as np

import google.genai as genai
from google.genai import types as gtypes

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)


DEFAULT_ANN_DIR  = os.path.join(ROOT, 'pgp-ego-finetuned', 'doScenes_repo', 'Annotations')
DEFAULT_KEYS     = os.path.join(HERE, 'Gemini_keys.txt')
DEFAULT_OUT      = os.path.join(ROOT, 'doscenes_gemini_embeddings.pkl')
MODEL_ID         = 'gemini-embedding-001'
OUT_DIM          = 768   # supported smaller output dim (default 3072)

PRIORITY = {'d': 0, 's': 1, 'sd': 2, 'ds': 3}


def load_doscenes_annotations(ann_dir):
    """Pool all annotator CSVs, return scene_num -> instruction (best by priority).
    Empty / missing -> not in dict."""
    by_scene = {}    # scene_num -> (priority, instruction)
    files = sorted(glob(os.path.join(ann_dir, '*.csv')))
    print(f'[anno] reading {len(files)} annotator csvs from {ann_dir}')
    for fp in files:
        with open(fp, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    scene_num = int(str(row.get('Scene Number', '')).strip())
                except Exception:
                    continue
                instr = (row.get('Instruction') or '').strip()
                if not instr:
                    continue
                ann_type = (row.get('Instruction Type') or '').strip().lower().rstrip(' "')
                pr = PRIORITY.get(ann_type, 9)
                cur = by_scene.get(scene_num)
                if (cur is None) or (pr < cur[0]):
                    by_scene[scene_num] = (pr, instr)
    return {sn: ins for sn, (_, ins) in by_scene.items()}


def find_working_keys(keys_file, model_id):
    """Pre-flight every API key against the embedding endpoint; keep all that work."""
    with open(keys_file) as f:
        all_keys = [k.strip() for k in f if k.strip()]
    print(f'[gemini] testing {len(all_keys)} keys for embeddings ...')
    working = []
    for i, k in enumerate(all_keys):
        try:
            c = genai.Client(api_key=k)
            c.models.embed_content(
                model=model_id, contents='test',
                config=gtypes.EmbedContentConfig(output_dimensionality=OUT_DIM),
            )
            working.append(k)
            print(f'  key {i:2d}: OK')
        except Exception as e:
            print(f'  key {i:2d}: FAIL {str(e)[:80]}')
        time.sleep(0.2)
    return working


def embed_batch_with_rotation(clients, key_last_call, model_id, texts, dim,
                              gap=60.0/14, retries=6):
    """Embed a batch using key rotation (least-recently-used) + retry on quota.

    The free-tier per-key quota for gemini-embedding-001 is small, so we have to
    spread calls across many keys and tolerate transient 429s gracefully.
    """
    for attempt in range(retries):
        now = time.time()
        idx = max(range(len(clients)), key=lambda i: now - key_last_call[i])
        elapsed = now - key_last_call[idx]
        if elapsed < gap:
            time.sleep(gap - elapsed)
        key_last_call[idx] = time.time()
        try:
            r = clients[idx].models.embed_content(
                model=model_id, contents=texts,
                config=gtypes.EmbedContentConfig(output_dimensionality=dim),
            )
            return [np.asarray(e.values, dtype=np.float32) for e in r.embeddings]
        except Exception as e:
            err = str(e)
            # On quota error, mark this key as recently used so rotator skips it.
            if '429' in err or 'quota' in err.lower():
                wait = 4.0
            else:
                wait = 1.5
            print(f'  retry {attempt+1} (key{idx}): {err[:120]}  (sleep {wait}s)')
            time.sleep(wait)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ann_dir',  default=DEFAULT_ANN_DIR)
    p.add_argument('--keys',     default=DEFAULT_KEYS)
    p.add_argument('--out',      default=DEFAULT_OUT)
    p.add_argument('--dim',      type=int, default=OUT_DIM)
    p.add_argument('--batch',    type=int, default=8)
    args = p.parse_args()

    if os.path.exists(args.out):
        print(f'[main] {args.out} already exists; reusing.')
        with open(args.out, 'rb') as f:
            d = pickle.load(f)
        print(f'  model={d["model"]}  dim={d["dim"]}  '
              f'instructions={len(d["instr_embeddings"])}  scenes={len(d["scene_to_instruction"])}')
        return

    scene_to_instruction = load_doscenes_annotations(args.ann_dir)
    print(f'  pooled annotations for {len(scene_to_instruction)} scenes')
    unique_instructions = sorted(set(scene_to_instruction.values()))
    print(f'  {len(unique_instructions)} unique instructions to embed')

    keys = find_working_keys(args.keys, MODEL_ID)
    if not keys:
        print('ERROR: no working keys for embeddings')
        sys.exit(2)
    print(f'[gemini] {len(keys)} working keys for {MODEL_ID}')
    clients = [genai.Client(api_key=k) for k in keys]
    key_last_call = [0.0] * len(clients)

    embeddings = {}
    t0 = time.time()
    gap = 60.0 / 8   # 8 RPM/key is a safer ceiling for embeddings free tier
    for start in range(0, len(unique_instructions), args.batch):
        batch = unique_instructions[start:start + args.batch]
        vecs = embed_batch_with_rotation(clients, key_last_call, MODEL_ID,
                                         batch, args.dim, gap=gap)
        if vecs is None:
            print(f'  FAILED batch starting at {start}; aborting')
            sys.exit(3)
        for txt, v in zip(batch, vecs):
            embeddings[txt] = v
        done = start + len(batch)
        if done % 32 == 0 or done == len(unique_instructions):
            elapsed = time.time() - t0
            rate = done / max(0.1, elapsed)
            eta = (len(unique_instructions) - done) / max(0.001, rate)
            print(f'  [{done:4d}/{len(unique_instructions)}]  '
                  f'elapsed={elapsed:.0f}s  rate={rate:.1f}/s  eta={eta:.0f}s')

    empty_emb = np.zeros(args.dim, dtype=np.float32)

    out = {
        'model':                MODEL_ID,
        'dim':                  args.dim,
        'instr_embeddings':     embeddings,
        'scene_to_instruction': scene_to_instruction,
        'empty_embedding':      empty_emb,
    }
    with open(args.out, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'\n[main] wrote {args.out}')
    print(f'  model={MODEL_ID}  dim={args.dim}')
    print(f'  unique instructions embedded: {len(embeddings)}')
    print(f'  scenes covered: {len(scene_to_instruction)}')


if __name__ == '__main__':
    main()
