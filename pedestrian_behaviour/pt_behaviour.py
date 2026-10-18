
"""
PTBehaviour — Full Model

Integrates the ST-GCN backbone with the two-stage Prospect Theory module
via a cross-attention gate.

Data flow:
    skeleton  ---> STGCNBackbone -----------------------------> H  (B, D)
                                                                   |
    pt_feats  ---> ProspectTheoryModule ---> utility U (B,)        |
                        |                     |                    |
                        |                 CrossAttnGate(H, U) ---->
                        |                                          |
                        |                                     classifier
                        |                                          |
                        +-- params (alpha,beta,lam,gamma) per pedestrian  P(cross)

General profile theta* is learned globally during training.
Individual profile theta_i = theta* + Delta_theta_i(clip) is computed per sample at inference.
"""

import torch
import torch.nn as nn

from stgcn_backbone import STGCNBackbone
from prospect_theory import ProspectTheoryModule


class CrossAttentionGate(nn.Module):
    """
    Modulates the visual embedding H using PT utility U.
    Gate = sigmoid(W [H; U_embed]) applied element-wise to H.

    U is a single scalar concatenated against a 256-dim H. Even with per-dimension
    weights of similar magnitude, H's 256 summed terms structurally outweigh U's lone
    term (~sqrt(256)=16x, for terms of comparable scale) in the linear layer's
    pre-activation, unless the network specifically learns a much larger weight for U to
    compensate -- which empirically it does not (see ANALYSIS_LOG.md Session 3, gate
    diagnostic). Projecting U into a small embedding first gives it a fairer footing to
    influence the gate, the same trick used to inject scalar conditioning signals (e.g.
    timestep) into embeddings elsewhere.
    """

    def __init__(self, embed_dim, utility_embed_dim=32):
        super().__init__()
        self.utility_proj = nn.Linear(1, utility_embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim + utility_embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, H, utility):
        U = utility.unsqueeze(-1)                          # (B, 1)
        U_embed = self.utility_proj(U)                      # (B, utility_embed_dim)
        return H * self.gate(torch.cat([H, U_embed], dim=-1))   # (B, D)


class PTBehaviour(nn.Module):
    """
    Args:
        in_channels   (int):   skeleton input channels (default 2: x, y)
        embed_dim     (int):   ST-GCN output dimension
        pt_input_dim  (int):   PT feature vector size
        pt_hidden_dim (int):   adaptation network hidden size
        num_joints    (int):   skeleton joints
        num_frames    (int):   observation window (60 = ~2s at 30fps)
        dropout       (float): dropout probability
    """

    def __init__(
        self,
        in_channels=2,
        embed_dim=256,
        pt_input_dim=5,
        pt_hidden_dim=32,
        num_joints=17,
        num_frames=60,
        dropout=0.2,
    ):
        super().__init__()

        self.backbone  = STGCNBackbone(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_joints=num_joints,
            num_frames=num_frames,
            dropout=dropout,
        )
        self.pt_module = ProspectTheoryModule(
            input_dim=pt_input_dim,
            hidden_dim=pt_hidden_dim,
        )
        self.gate = CrossAttentionGate(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, skeleton, pt_features, gain_value, loss_value, collision_prob):
        """
        Args:
            skeleton       (B, T, J, 2)
            pt_features    (B, 5)
            gain_value     (B,)
            loss_value     (B,)
            collision_prob (B,)

        Returns:
            logit   (B,)    raw crossing logit (apply sigmoid for probability)
            utility (B,)    PT utility score
            params  dict    individual CPT params per pedestrian
            general dict    population-level general profile theta*
        """
        H                    = self.backbone(skeleton)
        utility, params, gp  = self.pt_module(
            pt_features, gain_value, loss_value, collision_prob)
        H_gated              = self.gate(H, utility)
        logit                = self.classifier(H_gated).squeeze(-1)
        return logit, utility, params, gp

    def predict_proba(self, skeleton, pt_features, gain_value, loss_value, collision_prob):
        """Returns P(crossing) in [0, 1]."""
        logit, utility, params, gp = self.forward(
            skeleton, pt_features, gain_value, loss_value, collision_prob)
        return torch.sigmoid(logit), utility, params, gp

    def get_general_profile(self):
        """Expose the learned population-level risk profile theta*."""
        return self.pt_module.get_general_profile()
