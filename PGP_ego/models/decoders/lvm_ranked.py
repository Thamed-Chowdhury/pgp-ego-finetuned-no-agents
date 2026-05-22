from models.decoders.decoder import PredictionDecoder
import torch
import torch.nn as nn
from typing import Dict, Union
from models.decoders.utils import cluster_traj


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LVMRanked(PredictionDecoder):
    """
    LVM decoder with a learnable confidence head over the K clustered trajectories.

    Same trajectory generation pipeline as LVM (sample num_samples latent codes,
    decode trajectories, K-means cluster into num_clusters modes). The cluster
    "1/rank" score from Ward's hierarchical merge is kept as an extra feature
    fed to a small confidence head, which produces learnable per-cluster
    log-probabilities. Training those log-probs with a rank-based cross-entropy
    loss teaches the model to assign highest confidence to the cluster closest
    to GT, second-highest to the next closest, and so on.
    """

    def __init__(self, args):
        super().__init__()
        self.agg_type = args['agg_type']
        self.num_samples = args['num_samples']
        self.op_len = args['op_len']
        self.lv_dim = args['lv_dim']
        self.encoding_size = args['encoding_size']
        self.hidden = nn.Linear(args['encoding_size'] + args['lv_dim'], args['hidden_size'])
        self.op_traj = nn.Linear(args['hidden_size'], args['op_len'] * 2)
        self.leaky_relu = nn.LeakyReLU()
        self.num_clusters = args['num_clusters']

        # Learnable confidence head over clusters.
        # Input per cluster: pooled scene context (encoding_size) + cluster
        # centroid trajectory (op_len*2) + heuristic cluster score (1).
        conf_in_dim = args['encoding_size'] + args['op_len'] * 2 + 1
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
                raise Exception('Expected ' + str(self.num_samples) +
                                ' encodings for each train/val data')

        # Cache pre-z encoding for the confidence head's global context.
        agg_encoding_pre_z = agg_encoding

        # Sample latent variable and concatenate with aggregated encoding.
        batch_size = agg_encoding.shape[0]
        z = torch.randn(batch_size, self.num_samples, self.lv_dim, device=device)
        agg_encoding_z = torch.cat((agg_encoding, z), dim=2)
        h = self.leaky_relu(self.hidden(agg_encoding_z))

        # Output trajectories.
        traj = self.op_traj(h)
        traj = traj.reshape(batch_size, self.num_samples, self.op_len, 2)

        # Cluster into K modes (and get the heuristic 1/rank scores).
        traj_clustered, cluster_scores = cluster_traj(self.num_clusters, traj)

        # Learnable confidence head.
        # Global scene context: mean of per-sample encodings.
        global_ctx = agg_encoding_pre_z.mean(dim=1)  # (B, encoding_size)
        global_ctx_rep = global_ctx.unsqueeze(1).expand(-1, self.num_clusters, -1)
        traj_flat = traj_clustered.reshape(batch_size, self.num_clusters, -1)
        cluster_scores_rep = cluster_scores.unsqueeze(-1).float()
        conf_in = torch.cat([global_ctx_rep, traj_flat, cluster_scores_rep], dim=-1)

        conf_h = self.leaky_relu(self.conf_h1(conf_in))
        conf_h = self.leaky_relu(self.conf_h2(conf_h))
        logits = self.conf_op(conf_h).squeeze(-1)  # (B, K)
        log_probs = self.log_softmax(logits)       # (B, K)

        predictions = {'traj': traj_clustered, 'probs': log_probs}

        # Pass through remaining encoder/aggregator outputs (e.g. 'pi' for pi_bc).
        if isinstance(inputs, dict):
            for key, val in inputs.items():
                if key != 'agg_encoding':
                    predictions[key] = val

        # Keep the heuristic cluster scores for diagnostics if downstream wants them.
        predictions['cluster_scores'] = cluster_scores

        return predictions
