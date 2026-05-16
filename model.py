# ============================================================
# model.py
# ============================================================

import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch_geometric.nn import GATConv

# ============================================================
# GRAPH ATTENTION NETWORK
# ============================================================

class RoofGNN(torch.nn.Module):

    def __init__(self):

        super().__init__()

        # Input features:
        # [x, y, z, visibility]
        in_channels = 4

        # ====================================================
        # GNN Layers
        # ====================================================

        self.conv1 = GATConv(
            in_channels,
            64,
            heads=4
        )

        self.conv2 = GATConv(
            64 * 4,
            128,
            heads=4
        )

        self.conv3 = GATConv(
            128 * 4,
            128,
            heads=2
        )

        # ====================================================
        # MLP Decoder
        # ====================================================

        self.lin1 = Linear(
            128 * 2,
            128
        )

        self.lin2 = Linear(
            128,
            64
        )

        self.lin3 = Linear(
            64,
            2
        )

    def forward(
        self,
        x,
        edge_index
    ):

        # ====================================================
        # GNN
        # ====================================================

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.conv3(
            x,
            edge_index
        )

        x = F.relu(x)

        # ====================================================
        # DECODER
        # ====================================================

        x = self.lin1(x)

        x = F.relu(x)

        x = self.lin2(x)

        x = F.relu(x)

        x = self.lin3(x)

        return x