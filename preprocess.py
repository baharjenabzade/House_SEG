# ============================================================
# preprocess.py
# NORMALIZED VERSION
# ============================================================

import os
import json
import torch
import numpy as np

from tqdm import tqdm
from torch_geometric.data import Data

# ============================================================
# CONFIG
# ============================================================

RAW_DATASET_DIR = "dataset"

PROCESSED_DIR = "processed"

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)

# ============================================================
# CLEAN ARRAYS
# ============================================================

def clean_array(arr):

    arr = np.array(
        arr,
        dtype=np.float32
    )

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return arr

# ============================================================
# NORMALIZATION
# ============================================================

def compute_normalization(coords):

    coords = clean_array(coords)

    mean = coords.mean(
        axis=0,
        keepdims=True
    )

    std = coords.std(
        axis=0,
        keepdims=True
    )

    std = std + 1e-6

    return mean, std

def apply_normalization(
    coords,
    mean,
    std
):

    coords = clean_array(coords)

    coords = (
        coords - mean
    ) / std

    return coords

# ============================================================
# VALIDATE JSON
# ============================================================

def is_valid_roof_json(data):

    if not isinstance(data, dict):
        return False

    required = [
        "ortho",
        "perspective_1"
    ]

    for key in required:

        if key not in data:
            return False

    return True

# ============================================================
# FIND JSON FILES
# ============================================================

def find_valid_json_files(root_dir):

    valid_jsons = []

    for root, dirs, files in os.walk(root_dir):

        for file in files:

            if not file.endswith(".json"):
                continue

            # skip visualization/debug files
            bad_keywords = [
                "colored",
                "output",
                "preview",
                "render",
                "visualization"
            ]

            if any(
                k in file.lower()
                for k in bad_keywords
            ):
                continue

            full_path = os.path.join(
                root,
                file
            )

            try:

                with open(full_path, "r") as f:

                    data = json.load(f)

                if is_valid_roof_json(data):

                    valid_jsons.append(
                        full_path
                    )

            except:
                continue

    return valid_jsons

# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_node_features(perspective):

    nodes_xyz = perspective[
        "nodes_xyz"
    ]

    # ========================================================
    # NORMALIZE
    # ========================================================

    nodes_xyz = compute_normalization(
        nodes_xyz
    )

    x = torch.tensor(
        nodes_xyz,
        dtype=torch.float
    )

    # ========================================================
    # VISIBILITY FEATURE
    # ========================================================

    num_nodes = x.shape[0]

    visibility = np.ones(
        num_nodes,
        dtype=np.float32
    )

    unmatched = perspective.get(
        "unmatched_nodes",
        []
    )

    for idx in unmatched:

        if idx < num_nodes:

            visibility[idx] = 0.0

    visibility = torch.tensor(
        visibility,
        dtype=torch.float
    ).unsqueeze(1)

    # ========================================================
    # CONCAT FEATURES
    # ========================================================

    x = torch.cat(
        [x, visibility],
        dim=1
    )

    return x

# ============================================================
# EXTRACT EDGES
# ============================================================

def extract_edges(perspective):

    edges = perspective[
        "node_idx_edges"
    ]

    edge_index = torch.tensor(
        edges,
        dtype=torch.long
    ).t().contiguous()

    return edge_index

# ============================================================
# EXTRACT TARGETS
# ============================================================

def extract_targets(ortho):

    ortho_nodes = ortho[
        "nodes_xyz"
    ]

    # ========================================================
    # NORMALIZE TARGETS
    # ========================================================

    ortho_nodes = compute_normalization(
        ortho_nodes
    )

    # predict only x,y
    ortho_xy = ortho_nodes[:, :2]

    y = torch.tensor(
        ortho_xy,
        dtype=torch.float
    )

    return y

# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph(json_path):

    with open(json_path, "r") as f:

        roof = json.load(f)

    perspective = roof["perspective_1"]

    ortho = roof["ortho"]

    # ========================================================
    # RAW COORDS
    # ========================================================

    perspective_xyz = clean_array(
        perspective["nodes_xyz"]
    )

    ortho_xyz = clean_array(
        ortho["nodes_xyz"]
    )

    # ========================================================
    # SHARED NORMALIZATION
    # ========================================================

    mean, std = compute_normalization(
        perspective_xyz
    )

    perspective_xyz = apply_normalization(
        perspective_xyz,
        mean,
        std
    )

    ortho_xyz = apply_normalization(
        ortho_xyz,
        mean,
        std
    )

    # ========================================================
    # INPUT FEATURES
    # ========================================================

    x = torch.tensor(
        perspective_xyz,
        dtype=torch.float
    )

    # ========================================================
    # VISIBILITY FEATURE
    # ========================================================

    num_nodes = x.shape[0]

    visibility = np.ones(num_nodes)

    unmatched = perspective.get(
        "unmatched_nodes",
        []
    )

    for idx in unmatched:

        if idx < num_nodes:
            visibility[idx] = 0

    visibility = torch.tensor(
        visibility,
        dtype=torch.float
    ).unsqueeze(1)

    x = torch.cat(
        [x, visibility],
        dim=1
    )

    # ========================================================
    # EDGES
    # ========================================================

    edge_index = extract_edges(
        perspective
    )

    # ========================================================
    # TARGETS
    # ========================================================

    ortho_xy = ortho_xyz[:, :2]

    y = torch.tensor(
        ortho_xy,
        dtype=torch.float
    )

    # ========================================================
    # GRAPH
    # ========================================================

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )

    return data

# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset():

    json_files = find_valid_json_files(
        RAW_DATASET_DIR
    )

    print(
        f"\nFound {len(json_files)} valid roofs"
    )

    for idx, json_path in enumerate(
        tqdm(json_files)
    ):

        try:

            print(
                f"\nProcessing: {json_path}"
            )

            graph = build_graph(
                json_path
            )

            filename = (
                f"roof_{idx:06d}.pt"
            )

            save_path = os.path.join(
                PROCESSED_DIR,
                filename
            )

            torch.save(
                graph,
                save_path
            )

        except Exception as e:

            print(
                f"\nERROR: {json_path}"
            )

            print(e)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    process_dataset()

    print(
        "\nPreprocessing Complete!"
    )