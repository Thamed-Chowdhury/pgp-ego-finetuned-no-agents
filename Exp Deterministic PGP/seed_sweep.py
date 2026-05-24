"""
Seed sweep for the deterministic LVMRankedText inference.

Iterates a list (or range) of seeds, runs the full no-agents test eval for
each, and records ADE@6s + other metrics into a JSON log. Stops early as soon
as a seed produces ADE@6s <= --target (default 2.65 m).

Layout under --out_root (default: results/seed_sweep/):
    seed_<S>/inference_cache.pkl
    seed_<S>/submission.csv
    seed_<S>/self_eval_metrics.json
    seed_<S>/seed.json
And a sweep-level log:
    results/seed_sweep/sweep_log.json   # appended after every seed
    results/seed_sweep/sweep_summary.csv

Usage:
    python3 seed_sweep.py --seeds 0,1,2,3,4 --target 2.65
    python3 seed_sweep.py --range 0:50      --target 2.65
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def parse_seeds(args):
    if args.range:
        a, b = args.range.split(':')
        return list(range(int(a), int(b)))
    if args.seeds:
        return [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    raise SystemExit('must pass --seeds or --range')


def run_one_seed(seed, out_dir, batch_size, num_samples, log_path):
    cmd = [
        'python3', '-u', os.path.join(HERE, 'run_deterministic_inference.py'),
        '--config',         os.path.join(HERE, 'configs', 'pgp_ego_gatx2_lvm_ranked_text_stage3.yml'),
        '--test_root',      os.path.join(ROOT, 'nuscenes_data', 'v1-test'),
        '--trainval_stats', os.path.join(HERE, 'data', 'stats.pickle'),
        '--test_preproc',   os.path.join(HERE, 'data', 'test_preproc'),
        '--checkpoint',     os.path.join(HERE, 'checkpoints', 'stage3_best.tar'),
        '--doscenes_repo',  os.path.join(ROOT, 'pgp-ego-finetuned', 'doScenes_repo'),
        '--text_emb_pkl',   os.path.join(HERE, 'data', 'doscenes_gemini_embeddings.pkl'),
        '--out_dir',        out_dir,
        '--seed',           str(seed),
        '--batch_size',     str(batch_size),
        '--num_samples',    str(num_samples),
    ]
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(ROOT, 'PGP_ego') + ':' + env.get('PYTHONPATH', '')
    t0 = time.time()
    with open(log_path, 'w') as logf:
        proc = subprocess.run(
            cmd, cwd=os.path.join(ROOT, 'PGP_ego'),
            env=env, stdout=logf, stderr=subprocess.STDOUT,
        )
    dt = time.time() - t0
    if proc.returncode != 0:
        return None, dt
    metrics_path = os.path.join(out_dir, 'self_eval_metrics.json')
    if not os.path.isfile(metrics_path):
        return None, dt
    with open(metrics_path) as f:
        agg = json.load(f)['aggregate']
    return agg, dt


def append_sweep_log(log_path, entry):
    if os.path.isfile(log_path):
        with open(log_path) as f:
            data = json.load(f)
    else:
        data = []
    data.append(entry)
    with open(log_path, 'w') as f:
        json.dump(data, f, indent=2)


def append_csv(csv_path, entry, header):
    new = not os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow([entry.get(k, '') for k in header])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', help='comma-separated list, e.g. 0,1,2,7')
    p.add_argument('--range', help='start:stop (exclusive), e.g. 0:50')
    p.add_argument('--out_root', default=os.path.join(HERE, 'results', 'seed_sweep'))
    p.add_argument('--target', type=float, default=2.65,
                   help='Stop the sweep as soon as ADE@6s <= this.')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_samples', type=int, default=1000)
    p.add_argument('--keep_going', action='store_true',
                   help='Continue sweeping even after the target is met.')
    args = p.parse_args()

    seeds = parse_seeds(args)
    os.makedirs(args.out_root, exist_ok=True)
    os.makedirs(os.path.join(HERE, 'logs'), exist_ok=True)
    log_path = os.path.join(args.out_root, 'sweep_log.json')
    csv_path = os.path.join(args.out_root, 'sweep_summary.csv')
    header = ['seed', 'ade_2s', 'ade_4s', 'ade_6s', 'fde', 'miss_rate',
              'offroad', 'offroad_rate', 'wall_s']

    best = {'seed': None, 'ade_6s': float('inf')}
    print(f'[sweep] {len(seeds)} seeds, target ADE@6s <= {args.target}')
    for seed in seeds:
        out_dir = os.path.join(args.out_root, f'seed_{seed}')
        os.makedirs(out_dir, exist_ok=True)
        per_seed_log = os.path.join(HERE, 'logs', f'sweep_seed_{seed}.log')
        print(f'[sweep] seed={seed}  out={out_dir}  log={per_seed_log}')
        agg, dt = run_one_seed(seed, out_dir, args.batch_size, args.num_samples, per_seed_log)
        if agg is None:
            print(f'[sweep] seed={seed} FAILED (see {per_seed_log})')
            entry = {'seed': seed, 'ade_6s': None, 'wall_s': round(dt, 1),
                     'note': 'failed'}
            append_sweep_log(log_path, entry)
            continue

        entry = {'seed': seed, 'wall_s': round(dt, 1), **{k: agg[k] for k in
                 ['ade_2s', 'ade_4s', 'ade_6s', 'fde', 'miss_rate',
                  'offroad', 'offroad_rate']}}
        append_sweep_log(log_path, entry)
        append_csv(csv_path, entry, header)
        print(f'[sweep] seed={seed:>4d}  ADE@6s={agg["ade_6s"]:.6f}  '
              f'ADE@4s={agg["ade_4s"]:.4f}  ADE@2s={agg["ade_2s"]:.4f}  '
              f'wall={dt:.1f}s')

        if agg['ade_6s'] < best['ade_6s']:
            best = {'seed': seed, **{k: agg[k] for k in
                    ['ade_2s', 'ade_4s', 'ade_6s', 'fde', 'miss_rate',
                     'offroad', 'offroad_rate']}}
            with open(os.path.join(args.out_root, 'best_so_far.json'), 'w') as f:
                json.dump(best, f, indent=2)

        if agg['ade_6s'] <= args.target and not args.keep_going:
            print(f'\n[sweep] TARGET MET: seed={seed} ADE@6s={agg["ade_6s"]:.6f} <= {args.target}')
            break

    print('\n========== SWEEP DONE ==========')
    print(f'best_seed = {best["seed"]}')
    print(f'best_ade_6s = {best["ade_6s"]:.6f}')


if __name__ == '__main__':
    main()
