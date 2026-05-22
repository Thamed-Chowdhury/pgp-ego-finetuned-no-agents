"""
Text-conditioned confidence head over PGP's K=10 clustered trajectories.

Inputs per sample:
  - traj:    (K, T, 2)  — K candidate trajectories in agent frame
  - text_emb:(D_text,)  — frozen sentence embedding of the doScenes instruction

Output: (K,) logits, one score per trajectory. Softmax gives the confidence
distribution; argmax is the chosen trajectory.

Architecture is intentionally small — only this head trains; PGP is frozen.
"""

import torch
import torch.nn as nn


class TextConfidenceHead(nn.Module):
    def __init__(self, traj_len: int = 12, traj_dim: int = 2,
                 text_dim: int = 384, traj_feat_dim: int = 64,
                 hidden_dim: int = 128):
        super().__init__()
        self.traj_encoder = nn.Sequential(
            nn.Linear(traj_len * traj_dim, traj_feat_dim),
            nn.ReLU(),
            nn.Linear(traj_feat_dim, traj_feat_dim),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(traj_feat_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, traj: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        traj:     (B, K, T, 2)
        text_emb: (B, D_text)
        returns:  (B, K) logits
        """
        B, K, T, D = traj.shape
        traj_flat = traj.reshape(B, K, T * D)
        traj_feat = self.traj_encoder(traj_flat)             # (B, K, F)
        text_rep  = text_emb.unsqueeze(1).expand(-1, K, -1)  # (B, K, D_text)
        joint     = torch.cat([traj_feat, text_rep], dim=-1) # (B, K, F+D_text)
        logits    = self.scorer(joint).squeeze(-1)           # (B, K)
        return logits
