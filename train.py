# ============================================================
# train.py
# ============================================================

import os
import torch

from tqdm import tqdm

from dataloader import create_dataloader
from model import RoofGNN
from losses import total_loss

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8

EPOCHS = 300

LEARNING_RATE = 1e-3

SAVE_DIR = "checkpoints"

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

# ============================================================
# LOAD DATA
# ============================================================

train_loader = create_dataloader(
    "processed",
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ============================================================
# MODEL
# ============================================================

model = RoofGNN().to(DEVICE)

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ===========================================================
# SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10
)

# ============================================================
# TRAIN LOOP
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0

    loop = tqdm(train_loader)

    for batch in loop:

        batch = batch.to(DEVICE)

        optimizer.zero_grad()

        # ================================================
        # FORWARD
        # ================================================

        pred = model(
            batch.x,
            batch.edge_index
        )

        # ================================================
        # LOSS
        # ================================================

        loss = total_loss(
            pred,
            batch.y,
            batch.edge_index
        )

        # ================================================
        # BACKPROP
        # ================================================

        loss.backward()

        optimizer.step()
        

        epoch_loss += loss.item()

        loop.set_description(
            f"Epoch {epoch}"
        )

        loop.set_postfix(
            loss=loss.item()
        )

    avg_loss = epoch_loss / len(train_loader)

    print(
        f"\nEpoch {epoch} Average Loss: {avg_loss:.6f}"
    )
    scheduler.step(avg_loss)

    # ====================================================
    # SAVE CHECKPOINT
    # ====================================================

    save_path = os.path.join(
        SAVE_DIR,
        f"roof_gnn_epoch_{epoch}.pth"
    )

    torch.save(
        model.state_dict(),
        save_path
    )

# ============================================================
# FINAL SAVE
# ============================================================

torch.save(
    model.state_dict(),
    "roof_gnn_final.pth"
)

print("\nTraining Complete!")