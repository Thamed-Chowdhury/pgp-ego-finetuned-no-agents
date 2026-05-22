from metrics.metric import Metric
from typing import Dict, Union
import torch


class RankingXent(Metric):
    """
    Rank-aware soft cross-entropy over the K candidate trajectories.

    For each batch element we compute per-cluster ADE against the GT trajectory,
    sort clusters by ADE (rank 0 = closest), and build a target distribution
    that decreases monotonically with rank:

        target_rank(r) = (1 / (r + 1)) / H_K

    where H_K = sum_{i=1..K} 1/i is the K-th harmonic number. The closest
    trajectory gets the largest target weight, the 2nd closest the next, and so
    on. The loss is the cross-entropy between the model's predicted log-probs
    over clusters and this target distribution. This explicitly trains the
    confidence head to assign highest confidence to the trajectory nearest the
    GT, second-highest to the next nearest, etc.

    The target is detached so gradients flow only through log_probs (not via
    the ADE-based ranking signal).

    args (all optional):
        temperature: divide target weights by this temperature before
            re-normalising. Default 1.0 (use raw harmonic weights). Lower
            temperature → more peaked (closer to winner-take-all). Higher
            temperature → flatter.
        use_softmax_target: if True, use softmax(-ADE / temperature) as target
            instead of harmonic-rank weights. Default False.
    """

    def __init__(self, args: Dict = None):
        self.name = 'ranking_xent'
        self.temperature = 1.0
        self.use_softmax_target = False
        if args is not None:
            self.temperature = float(args.get('temperature', self.temperature))
            self.use_softmax_target = bool(args.get('use_softmax_target', self.use_softmax_target))

    def compute(self, predictions: Dict, ground_truth: Union[torch.Tensor, Dict]) -> torch.Tensor:
        traj = predictions['traj']               # (B, K, T, 2)
        log_probs = predictions['probs']         # (B, K) — expected to be log-softmax
        traj_gt = ground_truth['traj'] if isinstance(ground_truth, dict) else ground_truth

        batch_size, num_modes, seq_len, _ = traj.shape
        masks = ground_truth['masks'] if isinstance(ground_truth, dict) and 'masks' in ground_truth \
            else torch.zeros(batch_size, seq_len, device=traj.device)

        # Per-cluster ADE w.r.t. GT.
        traj_gt_rep = traj_gt.unsqueeze(1).expand(-1, num_modes, -1, -1)     # (B, K, T, 2)
        masks_rep = masks.unsqueeze(1).expand(-1, num_modes, -1)             # (B, K, T)
        err = traj_gt_rep - traj[..., :2]
        err = (err ** 2).sum(dim=3).sqrt()                                   # (B, K, T)
        valid = (1 - masks_rep)
        denom = valid.sum(dim=2).clamp(min=1.0)
        ade = (err * valid).sum(dim=2) / denom                               # (B, K)

        # Build target distribution.
        with torch.no_grad():
            if self.use_softmax_target:
                target = torch.softmax(-ade / self.temperature, dim=1)
            else:
                # Rank-based harmonic weights.
                _, sort_idx = torch.sort(ade, dim=1)         # ascending; sort_idx[b, r] = cluster at rank r
                ranks = torch.empty_like(ade)
                batch_idx = torch.arange(batch_size, device=ade.device).unsqueeze(1).expand(-1, num_modes)
                rank_values = torch.arange(num_modes, device=ade.device, dtype=ade.dtype).unsqueeze(0).expand(batch_size, -1)
                ranks.scatter_(1, sort_idx, rank_values)     # ranks[b, k] = rank of cluster k (0=closest)
                weights = 1.0 / (ranks + 1.0)
                weights = weights / self.temperature
                target = weights / weights.sum(dim=1, keepdim=True)

        # Soft cross-entropy: -sum_k target_k * log_probs_k, averaged over batch.
        loss = -(target * log_probs).sum(dim=1).mean()
        return loss
