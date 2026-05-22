"""
Text-conditioned variant of LVMRanked.

Same body as LVMRanked (sample 1000 z -> decode -> K-means -> learnable
confidence head over centroids). Adds a single new layer, `text_proj`, that
projects a Gemini sentence embedding to the model's encoding_size and
residual-adds it to the per-sample aggregated encoding before the latent z is
concatenated. Effect: the trajectory MLP (op_traj) AND the confidence head
both see the text signal.

Crucial design choice: `text_proj.weight` is initialised to zero, so at
fine-tune start the model is bit-for-bit identical to LVMRanked. Any
deviation only appears as gradients flow into text_proj. This makes
"--just-weights" fine-tuning from a non-text LVMRanked checkpoint safe (the
existing weights load 1:1; the new text_proj is the only zero-initialised
parameter).

Inputs dict additions (over LVMRanked):
    text_embedding: (B, text_dim) float tensor. Pass a zero vector for
        windows with no doScenes annotation - residual addition by zeros is
        a no-op so those windows behave exactly like LVMRanked.
"""
import torch
import torch.nn as nn
from typing import Dict, Union

from models.decoders.decoder import PredictionDecoder
from models.decoders.utils import cluster_traj


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LVMRankedText(PredictionDecoder):

    def __init__(self, args):
        super().__init__()
        self.agg_type = args['agg_type']
        self.num_samples = args['num_samples']
        self.op_len = args['op_len']
        self.lv_dim = args['lv_dim']
        self.encoding_size = args['encoding_size']
        self.text_dim = args['text_dim']

        self.hidden = nn.Linear(args['encoding_size'] + args['lv_dim'], args['hidden_size'])
        self.op_traj = nn.Linear(args['hidden_size'], args['op_len'] * 2)
        self.leaky_relu = nn.LeakyReLU()
        self.num_clusters = args['num_clusters']

        # Confidence head identical to LVMRanked.
        conf_in_dim = args['encoding_size'] + args['op_len'] * 2 + 1
        conf_hidden = args.get('conf_hidden_size', args['hidden_size'])
        self.conf_h1 = nn.Linear(conf_in_dim, conf_hidden)
        self.conf_h2 = nn.Linear(conf_hidden, conf_hidden)
        self.conf_op = nn.Linear(conf_hidden, 1)
        self.log_softmax = nn.LogSoftmax(dim=1)

        # NEW: text -> encoding_size residual.  Zero-init so the model starts
        # identical to LVMRanked when fine-tuning from an LVMRanked checkpoint.
        self.text_proj = nn.Linear(args['text_dim'], args['encoding_size'])
        nn.init.zeros_(self.text_proj.weight)
        nn.init.zeros_(self.text_proj.bias)

    def forward(self, inputs: Union[Dict, torch.Tensor]) -> Dict:
        if isinstance(inputs, torch.Tensor):
            agg_encoding = inputs
            text_emb = None
        else:
            agg_encoding = inputs['agg_encoding']
            text_emb = inputs.get('text_embedding', None)

        if self.agg_type == 'combined':
            agg_encoding = agg_encoding.unsqueeze(1).repeat(1, self.num_samples, 1)
        else:
            if len(agg_encoding.shape) != 3 or agg_encoding.shape[1] != self.num_samples:
                raise Exception('Expected ' + str(self.num_samples) +
                                ' encodings for each train/val data')

        # Text-conditioned residual: project to encoding_size and broadcast over samples.
        if text_emb is not None:
            # text_emb: (B, text_dim)  ->  (B, encoding_size)  ->  (B, 1, enc) broadcast
            t = self.text_proj(text_emb).unsqueeze(1)
            agg_encoding = agg_encoding + t

        # Cache (post-text) pre-z encoding for the confidence head.
        agg_encoding_pre_z = agg_encoding

        # Sample latent variable.
        batch_size = agg_encoding.shape[0]
        z = torch.randn(batch_size, self.num_samples, self.lv_dim, device=device)
        agg_encoding_z = torch.cat((agg_encoding, z), dim=2)
        h = self.leaky_relu(self.hidden(agg_encoding_z))

        # Output trajectories.
        traj = self.op_traj(h)
        traj = traj.reshape(batch_size, self.num_samples, self.op_len, 2)

        # Cluster into K modes.
        traj_clustered, cluster_scores = cluster_traj(self.num_clusters, traj)

        # Confidence head.
        global_ctx = agg_encoding_pre_z.mean(dim=1)                              # (B, enc)
        global_ctx_rep = global_ctx.unsqueeze(1).expand(-1, self.num_clusters, -1)
        traj_flat = traj_clustered.reshape(batch_size, self.num_clusters, -1)
        cluster_scores_rep = cluster_scores.unsqueeze(-1).float()
        conf_in = torch.cat([global_ctx_rep, traj_flat, cluster_scores_rep], dim=-1)

        conf_h = self.leaky_relu(self.conf_h1(conf_in))
        conf_h = self.leaky_relu(self.conf_h2(conf_h))
        logits = self.conf_op(conf_h).squeeze(-1)
        log_probs = self.log_softmax(logits)

        predictions = {'traj': traj_clustered, 'probs': log_probs}
        if isinstance(inputs, dict):
            for key, val in inputs.items():
                if key not in ('agg_encoding', 'text_embedding'):
                    predictions[key] = val
        predictions['cluster_scores'] = cluster_scores
        return predictions
