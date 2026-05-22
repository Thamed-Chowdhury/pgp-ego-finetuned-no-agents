"""
Training entry point for the "no surrounding agents" + ranking-confidence
experiment.

This script subclasses the stock Trainer to zero the surrounding-agent
feature tensors in every minibatch (train and val) before they reach the
model. The original train.py / Trainer remain untouched so any concurrent
experiment using them is unaffected.

Usage:
    python train_no_agents.py \
        -c configs/pgp_gatx2_lvm_no_agents.yml \
        -r /teamspace/studios/this_studio/nuscenes_data \
        -d /teamspace/studios/this_studio/pgp_preprocessed \
        -o /teamspace/studios/this_studio/pgp_output_no_agents_ranked \
        -n 50
"""

import argparse
import os
import time
import math

import torch
import torch.utils.data as torch_data
import yaml
from torch.utils.tensorboard import SummaryWriter

from train_eval.trainer import Trainer
import train_eval.utils as u
from no_agents_utils import zero_agents_in_inputs


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TrainerNoAgents(Trainer):
    """Trainer that zeros surrounding-agent features in every minibatch."""

    def run_epoch(self, mode: str, dl: torch_data.DataLoader):
        if mode == 'val':
            self.model.eval()
        else:
            self.model.train()

        epoch_metrics = self.initialize_metrics_for_epoch(mode)

        st_time = time.time()
        for i, data in enumerate(dl):
            data = u.send_to_device(u.convert_double_to_float(data))

            # >>> No-agents intervention: zero vehicle + pedestrian features.
            data['inputs'] = zero_agents_in_inputs(data['inputs'])

            predictions = self.model(data['inputs'])

            if mode == 'train':
                loss = self.compute_loss(predictions, data['ground_truth'])
                self.back_prop(loss)

            minibatch_time = time.time() - st_time
            st_time = time.time()

            minibatch_metrics, epoch_metrics = self.aggregate_metrics(
                epoch_metrics, minibatch_time, predictions, data['ground_truth'], mode)

            if mode == 'train':
                self.log_tensorboard_train(minibatch_metrics)

            if i % self.log_period == self.log_period - 1:
                self.print_metrics(epoch_metrics, dl, mode)

        if mode == 'val':
            self.log_tensorboard_val(epoch_metrics)

        return epoch_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--data_root", required=True)
    parser.add_argument("-d", "--data_dir", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument("-n", "--num_epochs", required=True)
    parser.add_argument("-w", "--checkpoint", required=False)
    parser.add_argument("--just-weights", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'tensorboard_logs'), exist_ok=True)

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard_logs'))

    trainer = TrainerNoAgents(cfg, args.data_root, args.data_dir,
                              checkpoint_path=args.checkpoint,
                              just_weights=args.just_weights, writer=writer)
    trainer.train(num_epochs=int(args.num_epochs), output_dir=args.output_dir)
    writer.close()


if __name__ == '__main__':
    main()
