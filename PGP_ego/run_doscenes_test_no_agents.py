"""
Run the no-agents Stage-2 (ego-finetuned) PGP checkpoint on the nuScenes
v1.0-test split via the existing doScenes pipeline, with surrounding-agent
feature tensors zeroed at every inference step.

This re-uses stage_preprocess / stage_submission / stage_self_eval from
run_doscenes_test_baseline.py and only re-implements stage_inference so the
model's inputs have agents zeroed before each forward pass. The original
run_doscenes_test_baseline.py is not modified (other experiments use it).

Usage:
    python run_doscenes_test_no_agents.py \
        --config       configs/pgp_ego_no_agents_stage2.yml \
        --test_root    /teamspace/studios/this_studio/nuscenes_data/v1-test \
        --trainval_stats /teamspace/studios/this_studio/pgp_ego_preprocessed/stats.pickle \
        --test_preproc /teamspace/studios/this_studio/pgp_ego_test_preprocessed \
        --checkpoint   /teamspace/studios/this_studio/pgp_output_no_agents_ranked_stage2/checkpoints/best.tar \
        --doscenes_repo /teamspace/studios/this_studio/pgp-ego-finetuned/doScenes_repo \
        --out_dir      /teamspace/studios/this_studio/test_no_agents_results
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Mirror the parent script's monkey-patch for v1.0-test (the prediction
# challenge split files don't include test, so the NuScenesTrajectories
# parent __init__ would otherwise fail).
import nuscenes.eval.prediction.splits as _ns_pred_splits
_ns_pred_splits.get_prediction_challenge_split = lambda split, dataroot=None: []

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.nuScenes.nuScenes_ego_graphs_doscenes import (
    NuScenesEgoGraphsDoScenes, HISTORY_LEN, FUTURE_LEN, MIN_SCENE_SAMPLES,
)
from train_eval.initialization import initialize_prediction_model
import train_eval.utils as u

from run_doscenes_test_baseline import (
    make_test_helper, scenes_used_for_eval,
    stage_preprocess, stage_submission, stage_self_eval,
)
from no_agents_utils import zero_agents_in_inputs

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def stage_inference_no_agents(cfg, helper, test_dir, checkpoint, batch_size=4,
                              num_workers=0, num_samples=100):
    """Same as stage_inference in run_doscenes_test_baseline but zeroes agent
    features before each model forward pass."""
    print(f'[inference] device = {device}')
    model = initialize_prediction_model(
        cfg['encoder_type'], cfg['aggregator_type'], cfg['decoder_type'],
        cfg['encoder_args'], cfg['aggregator_args'], cfg['decoder_args'],
    ).float().to(device)
    model.eval()
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.aggregator.num_samples = num_samples
    model.decoder.num_samples = num_samples
    print(f'[inference] checkpoint loaded: {checkpoint}')

    sa = dict(cfg['test_set_args'])
    sa['random_flips'] = False
    sa['split'] = 'doscenes_test'

    ds = NuScenesEgoGraphsDoScenes('load_data', test_dir, sa, helper)
    tokens = [t.split('_', 1)[1] for t in ds.token_list]
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    out = {}
    cursor = 0
    with torch.no_grad():
        for bi, data in enumerate(dl):
            data = u.send_to_device(u.convert_double_to_float(data))
            data['inputs'] = zero_agents_in_inputs(data['inputs'])
            preds = model(data['inputs'])
            traj = preds['traj'].detach().cpu().numpy()
            probs = preds['probs'].detach().cpu().numpy()
            gt = data['ground_truth']['traj'].detach().cpu().numpy()
            for b in range(traj.shape[0]):
                top_idx = int(np.argmax(probs[b]))
                out[tokens[cursor + b]] = {
                    'top_idx':      top_idx,
                    'top_traj_pgp': traj[b, top_idx].astype(np.float32),
                    'all_traj_pgp': traj[b].astype(np.float32),
                    'probs':        probs[b].astype(np.float32),
                    'gt_pgp':       gt[b].astype(np.float32),
                }
            cursor += traj.shape[0]
            if (bi + 1) % 5 == 0 or cursor == len(ds):
                print(f'  infer batch {bi+1}/{len(dl)}  ({cursor}/{len(ds)})')
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--test_root', required=True)
    p.add_argument('--trainval_stats', required=True)
    p.add_argument('--test_preproc', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--doscenes_repo', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--num_samples', type=int, default=100)
    p.add_argument('--skip_preprocess', action='store_true')
    p.add_argument('--skip_inference', action='store_true')
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg['version'] = 'v1.0-test'

    helper = make_test_helper(args.test_root)
    nusc = helper.data
    print(f'[main] {len(nusc.scene)} test scenes loaded')

    scene_tuples = scenes_used_for_eval(nusc)
    print(f'[main] {len(scene_tuples)} scenes meet >= {MIN_SCENE_SAMPLES} samples')

    if not args.skip_preprocess:
        stage_preprocess(cfg, helper, args.test_preproc, args.trainval_stats,
                         batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        print('[main] skipping preprocessing per --skip_preprocess')

    os.makedirs(args.out_dir, exist_ok=True)
    inf_cache = os.path.join(args.out_dir, 'inference_cache.pkl')
    if args.skip_inference and os.path.isfile(inf_cache):
        import pickle
        with open(inf_cache, 'rb') as f:
            out = pickle.load(f)
        print(f'[main] loaded cached inference: {len(out)} entries')
    else:
        out = stage_inference_no_agents(
            cfg, helper, args.test_preproc, args.checkpoint,
            batch_size=args.batch_size, num_workers=args.num_workers,
            num_samples=args.num_samples)
        import pickle
        with open(inf_cache, 'wb') as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'[main] cached inference -> {inf_cache}')

    submission_csv = os.path.join(args.out_dir, 'submission.csv')
    stage_submission(out, scene_tuples, submission_csv)

    metrics_json = os.path.join(args.out_dir, 'self_eval_metrics.json')
    agg, _ = stage_self_eval(out, scene_tuples, helper, args.test_root,
                             args.doscenes_repo, metrics_json)

    print('\n=== v1.0-test results (no-agents PGP, stage-2 ego-finetune) ===')
    print(f'   N scenes evaluated: {len([t for t in scene_tuples if t[0] in out])}')
    for k in ['ade_2s', 'ade_4s', 'ade_6s', 'fde', 'miss_rate',
              'speed_error', 'ahe', 'fhe', 'offroad', 'offroad_rate', 'offyaw']:
        print(f'   {k:14s} = {agg[k]:.4f}')
    print('=================================================================')


if __name__ == '__main__':
    main()
