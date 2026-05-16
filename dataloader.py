# ============================================================
# dataset.py
# ============================================================

import os
import torch

from torch_geometric.loader import DataLoader

# ============================================================
# DATASET
# ============================================================

class RoofDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir):

        self.files = sorted([

            os.path.join(root_dir, f)

            for f in os.listdir(root_dir)

            if f.endswith(".pt")

        ])

        print(f"Loaded {len(self.files)} graphs")

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        graph = torch.load(
            self.files[idx],
            weights_only=False
        )

        return graph

# ============================================================
# DATALOADER
# ============================================================

def create_dataloader(
    root_dir,
    batch_size=4,
    shuffle=True
):

    dataset = RoofDataset(root_dir)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader