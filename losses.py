# ============================================================
# losses.py
# ============================================================

import torch
import torch.nn.functional as F

# ============================================================
# COORDINATE LOSS
# ============================================================

def coordinate_loss(pred, target):

    return F.mse_loss(pred, target)

# ============================================================
# EDGE LENGTH LOSS
# ============================================================

def edge_length_loss(
    pred,
    target,
    edge_index
):

    src = edge_index[0]
    dst = edge_index[1]

    pred_len = torch.norm(
        pred[src] - pred[dst],
        dim=1
    )

    target_len = torch.norm(
        target[src] - target[dst],
        dim=1
    )

    return F.mse_loss(
        pred_len,
        target_len
    )

# ============================================================
# ANGLE LOSS
# ============================================================

def angle_loss(
    pred,
    target,
    edge_index
):

    src = edge_index[0]
    dst = edge_index[1]

    pred_vec = pred[dst] - pred[src]
    target_vec = target[dst] - target[src]

    pred_vec = F.normalize(
        pred_vec,
        dim=1
    )

    target_vec = F.normalize(
        target_vec,
        dim=1
    )

    cosine = (
        pred_vec * target_vec
    ).sum(dim=1)

    target_cosine = torch.ones_like(cosine)

    return F.mse_loss(
        cosine,
        target_cosine
    )

# ============================================================
# TOTAL LOSS
# ============================================================

def total_loss(
    pred,
    target,
    edge_index
):

    coord = coordinate_loss(
        pred,
        target
    )

    edge = edge_length_loss(
        pred,
        target,
        edge_index
    )

    angle = angle_loss(
        pred,
        target,
        edge_index
    )

    loss = (
        coord
        + 0.5 * edge
        + 0.3 * angle
    )

    return loss