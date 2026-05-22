"""
LVMRankedRichCtx — same trajectory generator and ranking head idea as
LVMRanked, but with a richer feature set fed to the confidence head so it
can actually discriminate between the K=10 clusters.

What the original lvm_ranked confidence head saw, per cluster k:
    [ global_ctx (160) | traj_flat (24) | cluster_score (1) ]
  where global_ctx = mean over all num_samples (=1000) of agg_encoding. The
  per-sample variance — i.e. the structured information that distinguishes
  cluster A from cluster B — is averaged out before the head ever sees it.

What this decoder feeds, per cluster k:
    [ target_agent_enc (32)                ← explicit ego/target state
    | global_lane_ctx   (128)              ← mean over all 1000 sample
                                            attention outputs
    | per_cluster_lane_ctx (128)           ← mean only over samples assigned
                                            to cluster k (cluster-specific
                                            scene context)
    | traj_flat (24)                       ← the cluster centroid trajectory
    | cluster_score (1)                    ← heuristic 1/rank
    ] = 313 dims per cluster.

Recall (models/aggregators/pgp.py:127):
    agg_enc = concat( target_agent_encoding.repeat, att_op ) along last dim
            shape: (B, num_samples, 32 + 128 = 160)
So agg_encoding[:, :, :32] is the target-agent encoding (identical across
samples), and agg_encoding[:, :, 32:] is the per-sample attention output
from the PGP aggregator (which is what varies across the 1000 traversals).

Clustering: this decoder reuses cluster_and_rank from
models/decoders/utils.py (identical procedure, same Ray remote function) but
keeps the per-sample cluster labels so it can pool agg_encoding by cluster.
The existing LVMRanked decoder is untouched.
"""

from models.decoders.decoder import PredictionDecoder
import torch
import torch.nn as nn
from typing import Dict, Union
import ray

from models.decoders.utils import cluster_and_rank


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cluster_traj_with_lbls(k: int, traj: torch.Tensor):
    """
    Like cluster_traj() in models/decoders/utils.py, but also returns the
    per-sample cluster labels (so we can pool features per cluster).

    Returns:
        traj_clustered: (B, K, T, 2) — cluster centroid trajectories
        scores:         (B, K)      — heuristic 1/rank scores (Ward order)
        lbls:           (B, num_samples) long — cluster index per sample
    """
    batch_size = traj.shape[0]
    num_samples = traj.shape[1]
    traj_len = traj.shape[2]

    # Down-sample along time for faster clustering (matches utils.cluster_traj).
    data = traj[:, :, 0::3, :]
    data = data.reshape(batch_size, num_samples, -1).detach().cpu().numpy()

    cluster_ops = ray.get([cluster_and_rank.remote(k, data_slice) for data_slice in data])
    cluster_lbls = [c['lbls'] for c in cluster_ops]
    cluster_counts = [c['counts'] for c in cluster_ops]
    cluster_ranks = [c['ranks'] for c in cluster_ops]

    lbls_t = torch.as_tensor(cluster_lbls, device=device).long()                # (B, num_samples)
    lbls_full = lbls_t.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, traj_len, 2)    # (B, num_samples, T, 2)
    traj_summed = torch.zeros(batch_size, k, traj_len, 2, device=device).scatter_add(1, lbls_full, traj)
    cnt_tensor = torch.as_tensor(cluster_counts, device=device).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, traj_len, 2)
    traj_clustered = traj_summed / cnt_tensor

    scores = 1 / torch.as_tensor(cluster_ranks, device=device)
    scores = scores / torch.sum(scores, dim=1)[0]

    return traj_clustered, scores, lbls_t


def _pool_by_cluster(features: torch.Tensor, lbls: torch.Tensor,
                     num_clusters: int) -> torch.Tensor:
    """
    Per-cluster mean of per-sample features.

    Args:
        features:     (B, num_samples, D)
        lbls:         (B, num_samples) long, values in [0, num_clusters)
        num_clusters: K

    Returns:
        (B, K, D) — mean of features over samples in each cluster.
        Empty clusters (count == 0) get zero output.
    """
    B, S, D = features.shape
    K = num_clusters
    one_hot = torch.zeros(B, S, K, device=features.device)
    one_hot.scatter_(2, lbls.unsqueeze(-1), 1.0)                # (B, S, K)
    cnt = one_hot.sum(dim=1).clamp(min=1.0).unsqueeze(-1)       # (B, K, 1)
    summed = torch.einsum('bsk,bsd->bkd', one_hot, features)    # (B, K, D)
    return summed / cnt


class LVMRankedRichCtx(PredictionDecoder):
    """
    LVM decoder with a learnable confidence head whose inputs are enriched
    with target-agent encoding, global lane-graph context, and per-cluster
    pooled lane-graph context.

    Required decoder_args:
        agg_type, num_samples, op_len, lv_dim, encoding_size, hidden_size,
        num_clusters,
        conf_hidden_size (optional; default = hidden_size),
        target_agent_enc_size (optional; default 32) — first slice of
            agg_encoding's last dim, identical across samples in PGP.
    """

    def __init__(self, args):
        super().__init__()
        self.agg_type = args['agg_type']
        self.num_samples = args['num_samples']
        self.op_len = args['op_len']
        self.lv_dim = args['lv_dim']
        self.encoding_size = args['encoding_size']
        self.num_clusters = args['num_clusters']
        self.target_agent_enc_size = args.get('target_agent_enc_size', 32)
        self.lane_ctx_size = self.encoding_size - self.target_agent_enc_size

        # Trajectory MLP (same as lvm_ranked).
        self.hidden = nn.Linear(args['encoding_size'] + args['lv_dim'], args['hidden_size'])
        self.op_traj = nn.Linear(args['hidden_size'], args['op_len'] * 2)
        self.leaky_relu = nn.LeakyReLU()

        # Confidence head inputs (per cluster k):
        #   target_agent_enc       (target_agent_enc_size)
        #   global_lane_ctx        (lane_ctx_size)
        #   per_cluster_lane_ctx   (lane_ctx_size)
        #   traj_flat              (op_len * 2)
        #   cluster_score          (1)
        conf_in_dim = (self.target_agent_enc_size
                       + self.lane_ctx_size * 2
                       + args['op_len'] * 2
                       + 1)
        conf_hidden = args.get('conf_hidden_size', args['hidden_size'])
        self.conf_h1 = nn.Linear(conf_in_dim, conf_hidden)
        self.conf_h2 = nn.Linear(conf_hidden, conf_hidden)
        self.conf_op = nn.Linear(conf_hidden, 1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs: Union[Dict, torch.Tensor]) -> Dict:
        if isinstance(inputs, torch.Tensor):
            agg_encoding = inputs
        else:
            agg_encoding = inputs['agg_encoding']

        if self.agg_type == 'combined':
            agg_encoding = agg_encoding.unsqueeze(1).repeat(1, self.num_samples, 1)
        else:
            if len(agg_encoding.shape) != 3 or agg_encoding.shape[1] != self.num_samples:
                raise Exception(f'Expected {self.num_samples} encodings, '
                                f'got shape {tuple(agg_encoding.shape)}')

        # Cache the pre-z encoding for the confidence head.
        agg_encoding_pre_z = agg_encoding                    # (B, S, encoding_size)
        batch_size = agg_encoding.shape[0]

        # Sample z and decode trajectories (identical to lvm_ranked).
        z = torch.randn(batch_size, self.num_samples, self.lv_dim, device=device)
        h = self.leaky_relu(self.hidden(torch.cat((agg_encoding, z), dim=2)))
        traj = self.op_traj(h)
        traj = traj.reshape(batch_size, self.num_samples, self.op_len, 2)

        # Cluster trajectories AND keep per-sample labels.
        traj_clustered, cluster_scores, lbls = cluster_traj_with_lbls(
            self.num_clusters, traj)

        # ------- richer confidence-head features -------
        ta_size = self.target_agent_enc_size
        target_agent_enc = agg_encoding_pre_z[:, 0, :ta_size]                 # (B, 32) — identical across samples
        lane_per_sample = agg_encoding_pre_z[:, :, ta_size:]                  # (B, S, 128)

        global_lane_ctx = lane_per_sample.mean(dim=1)                         # (B, 128)
        per_cluster_lane_ctx = _pool_by_cluster(lane_per_sample, lbls,
                                                self.num_clusters)            # (B, K, 128)

        target_agent_rep = target_agent_enc.unsqueeze(1).expand(-1, self.num_clusters, -1)   # (B, K, 32)
        global_lane_rep = global_lane_ctx.unsqueeze(1).expand(-1, self.num_clusters, -1)     # (B, K, 128)
        traj_flat = traj_clustered.reshape(batch_size, self.num_clusters, -1)                # (B, K, 24)
        cluster_scores_rep = cluster_scores.unsqueeze(-1).float()                            # (B, K, 1)

        conf_in = torch.cat([target_agent_rep,
                             global_lane_rep,
                             per_cluster_lane_ctx,
                             traj_flat,
                             cluster_scores_rep], dim=-1)                                     # (B, K, conf_in_dim)

        conf_h = self.leaky_relu(self.conf_h1(conf_in))
        conf_h = self.leaky_relu(self.conf_h2(conf_h))
        logits = self.conf_op(conf_h).squeeze(-1)                                             # (B, K)
        log_probs = self.log_softmax(logits)                                                  # (B, K)

        predictions = {'traj': traj_clustered, 'probs': log_probs}
        if isinstance(inputs, dict):
            for key, val in inputs.items():
                if key != 'agg_encoding':
                    predictions[key] = val
        predictions['cluster_scores'] = cluster_scores
        return predictions
