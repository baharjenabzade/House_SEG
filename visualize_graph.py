# ============================================================
# visualize_graph.py
# ============================================================

import torch
import matplotlib.pyplot as plt

from dataloader import create_dataloader

# ============================================================
# LOAD DATA
# ============================================================

loader = create_dataloader(
    "processed",
    batch_size=1,
    shuffle=True
)

# ============================================================
# GET ONE GRAPH
# ============================================================

batch = next(iter(loader))

# ============================================================
# EXTRACT
# ============================================================

x = batch.x.numpy()

edge_index = batch.edge_index.numpy()

y = batch.y.numpy()

# ============================================================
# INPUT GRAPH (Perspective)
# ============================================================

plt.figure(figsize=(8, 8))

for edge in edge_index.T:

    i, j = edge

    x_coords = [x[i][0], x[j][0]]
    y_coords = [x[i][1], x[j][1]]

    plt.plot(
        x_coords,
        y_coords,
        'b-'
    )

plt.scatter(
    x[:, 0],
    x[:, 1]
)

plt.title("Perspective Graph")

plt.axis("equal")

plt.show()

# ============================================================
# TARGET GRAPH (Orthographic)
# ============================================================

plt.figure(figsize=(8, 8))

for edge in edge_index.T:

    i, j = edge

    x_coords = [y[i][0], y[j][0]]
    y_coords = [y[i][1], y[j][1]]

    plt.plot(
        x_coords,
        y_coords,
        'r-'
    )

plt.scatter(
    y[:, 0],
    y[:, 1]
)

plt.title("Orthographic Target")

plt.axis("equal")

plt.show()