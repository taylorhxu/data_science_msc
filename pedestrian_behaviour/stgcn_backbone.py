"""
ST-GCN Backbone — adapted from ST CrossingPose (Zhang et al., IEEE TITS 2022)

Encodes a (T, J, 2) skeleton sequence into a (B, embed_dim) pedestrian state
embedding using spatial graph convolutions over the COCO 17-joint skeleton
followed by temporal 1D convolutions.

The embedding is passed to the CrossAttentionGate in pt_behaviour.py.
"""

import torch
import torch.nn as nn

# COCO 17-joint anatomical adjacency pairs
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]
NUM_JOINTS = 17


def build_adjacency(num_joints=NUM_JOINTS, edges=COCO_SKELETON):
    """Normalised symmetric adjacency matrix with self-loops."""
    A = torch.zeros(num_joints, num_joints)
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    A += torch.eye(num_joints)
    D = A.sum(dim=1, keepdim=True).clamp(min=1.0)
    return A / D


class SpatialGraphConv(nn.Module):
    """Single spatial GCN layer over joint nodes."""

    def __init__(self, in_ch, out_ch, num_joints=NUM_JOINTS):
        super().__init__()
        self.fc = nn.Linear(in_ch, out_ch)
        self.register_buffer('A', build_adjacency(num_joints))

    def forward(self, x):
        # x: (B, T, J, C)
        B, T, J, C = x.shape
        x = x.reshape(B * T, J, C)
        x = self.fc(x)                                      # (BT, J, out_ch)
        x = torch.bmm(self.A.unsqueeze(0).expand(B*T,-1,-1), x)
        return x.reshape(B, T, J, -1)


class STGCNBlock(nn.Module):
    """Spatial GCN + temporal conv + residual."""

    def __init__(self, in_ch, out_ch, num_joints=NUM_JOINTS,
                 t_kernel=3, dropout=0.1):
        super().__init__()
        self.spatial = SpatialGraphConv(in_ch, out_ch, num_joints)
        pad = (t_kernel - 1) // 2
        self.temporal = nn.Conv1d(out_ch, out_ch, kernel_size=t_kernel, padding=pad)
        self.norm     = nn.LayerNorm(out_ch)
        self.drop     = nn.Dropout(dropout)
        self.relu     = nn.ReLU()
        self.residual = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # x: (B, T, J, C_in)
        B, T, J, _ = x.shape
        res = self.residual(x)
        out = self.relu(self.spatial(x))                    # (B, T, J, out_ch)
        # temporal conv per joint
        out = out.permute(0, 2, 3, 1).reshape(B*J, -1, T) # (BJ, out_ch, T)
        out = self.temporal(out)
        out = out.reshape(B, J, -1, T).permute(0, 3, 1, 2) # (B, T, J, out_ch)
        return self.drop(self.norm(out + res))


class STGCNBackbone(nn.Module):
    """
    Three-block ST-GCN producing a (B, embed_dim) pedestrian state embedding.

    Args:
        in_channels (int):   input features per joint (2 for x, y)
        embed_dim   (int):   output embedding dimension
        num_joints  (int):   skeleton joints
        num_frames  (int):   observation window length
        dropout     (float): dropout probability
    """

    def __init__(self, in_channels=2, embed_dim=256,
                 num_joints=NUM_JOINTS, num_frames=60, dropout=0.1):
        super().__init__()
        self.input_bn = nn.BatchNorm2d(in_channels)
        self.blocks   = nn.ModuleList([
            STGCNBlock(in_channels, 64,       num_joints, dropout=dropout),
            STGCNBlock(64,          128,      num_joints, dropout=dropout),
            STGCNBlock(128,         embed_dim,num_joints, dropout=dropout),
        ])
        self.embed_dim = embed_dim

    def forward(self, skeleton):
        """
        Args:
            skeleton (B, T, J, 2)
        Returns:
            H (B, embed_dim)
        """
        B, T, J, C = skeleton.shape
        x = skeleton.permute(0, 3, 1, 2)   # (B, C, T, J)
        x = self.input_bn(x)
        x = x.permute(0, 2, 3, 1)          # (B, T, J, C)
        for block in self.blocks:
            x = block(x)
        return x.mean(dim=[1, 2])           # global avg pool -> (B, embed_dim)
