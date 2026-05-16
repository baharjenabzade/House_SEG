# ============================================================
# inference.py
# ============================================================

import torch
import matplotlib.pyplot as plt

from dataloader import create_dataloader
from model import RoofGNN

# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# LOAD MODEL
# ============================================================

model = RoofGNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        "roof_gnn_final.pth",
        map_location=DEVICE
    )
)

model.eval()

# ============================================================
# LOAD SAMPLE
# ============================================================

loader = create_dataloader(
    "processed",
    batch_size=1,
    shuffle=True
)

batch = next(iter(loader))

batch = batch.to(DEVICE)

# ============================================================
# MODEL PREDICTION
# ============================================================

with torch.no_grad():

    pred = model(
        batch.x,
        batch.edge_index
    )

# ============================================================
# TO NUMPY
# ============================================================

# Perspective input
perspective = batch.x[:, :2].cpu().numpy()

# Ground truth ortho
gt = batch.y.cpu().numpy()

# Predicted ortho
pred = pred.cpu().numpy()

# Graph edges
edges = batch.edge_index.cpu().numpy()

# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)

# ============================================================
# 1. PERSPECTIVE INPUT
# ============================================================

ax = axes[0]

for edge in edges.T:

    i, j = edge

    ax.plot(
        [perspective[i][0], perspective[j][0]],
        [perspective[i][1], perspective[j][1]]
    )

ax.scatter(
    perspective[:, 0],
    perspective[:, 1]
)

ax.set_title("Original Perspective Input")

ax.axis("equal")

# ============================================================
# 2. GROUND TRUTH
# ============================================================

ax = axes[1]

for edge in edges.T:

    i, j = edge

    ax.plot(
        [gt[i][0], gt[j][0]],
        [gt[i][1], gt[j][1]]
    )

ax.scatter(
    gt[:, 0],
    gt[:, 1]
)

ax.set_title("Ground Truth Orthographic")

ax.axis("equal")

# ============================================================
# 3. PREDICTION
# ============================================================

ax = axes[2]

for edge in edges.T:

    i, j = edge

    ax.plot(
        [pred[i][0], pred[j][0]],
        [pred[i][1], pred[j][1]]
    )

ax.scatter(
    pred[:, 0],
    pred[:, 1]
)

ax.set_title("Predicted Orthographic")

ax.axis("equal")

# ============================================================
# SHOW
# ============================================================

plt.tight_layout()

plt.show()