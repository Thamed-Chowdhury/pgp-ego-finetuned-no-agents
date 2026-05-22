from metrics.metric import Metric
from typing import Dict, Union
import torch


class Top1ADE(Metric):
    """
    ADE of the highest-confidence trajectory.

    Selects argmax over predicted probabilities (or log-probs — argmax order is
    the same) and returns the per-batch-averaged average displacement error of
    that single trajectory against the GT. This is the metric the deployed
    model is actually evaluated on (top-1 confidence), and it's what the
    rank-aware confidence training is designed to improve.
    """

    def __init__(self, args: Dict = None):
        self.name = 'top1_ade'

    def compute(self, predictions: Dict, ground_truth: Union[torch.Tensor, Dict]) -> torch.Tensor:
        traj = predictions['traj']            # (B, K, T, 2)
        probs = predictions['probs']          # (B, K)
        traj_gt = ground_truth['traj'] if isinstance(ground_truth, dict) else ground_truth

        batch_size, _, seq_len, _ = traj.shape
        masks = ground_truth['masks'] if isinstance(ground_truth, dict) and 'masks' in ground_truth \
            else torch.zeros(batch_size, seq_len, device=traj.device)

        top_idx = torch.argmax(probs, dim=1)                            # (B,)
        batch_idx = torch.arange(batch_size, device=traj.device)
        top_traj = traj[batch_idx, top_idx]                             # (B, T, 2)

        err = traj_gt - top_traj
        err = (err ** 2).sum(dim=2).sqrt()                              # (B, T)
        valid = (1 - masks)
        denom = valid.sum(dim=1).clamp(min=1.0)
        ade = (err * valid).sum(dim=1) / denom                          # (B,)
        return ade.mean()
