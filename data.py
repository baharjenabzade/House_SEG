import os
import zipfile
import shutil
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

# Your ZIP file
ZIP_FILE = "samples.zip"

# Temporary extraction folder
EXTRACT_DIR = "extracted_files"

# Final organized dataset folder
OUTPUT_DIR = "dataset"

# Dataset split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(EXTRACT_DIR, exist_ok=True)

train_dir = os.path.join(OUTPUT_DIR, "train")
val_dir = os.path.join(OUTPUT_DIR, "val")
test_dir = os.path.join(OUTPUT_DIR, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# ============================================================
# STEP 1: EXTRACT ZIP
# ============================================================

print("Extracting ZIP file...")

with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

print("Extraction complete!")

# ============================================================
# STEP 2: FIND ALL JSON + CSV FILES
# ============================================================

all_files = []

for root, dirs, files in os.walk(EXTRACT_DIR):

    for file in files:

        if file.endswith(".json") or file.endswith(".csv"):

            full_path = os.path.join(root, file)

            all_files.append(full_path)

print(f"Found {len(all_files)} files")

# ============================================================
# STEP 3: COLLECT ROOF IDS
# ============================================================

roof_ids = set()

for file_path in all_files:

    stem = Path(file_path).stem

    roof_ids.add(stem)

roof_ids = sorted(list(roof_ids))

print(f"Found {len(roof_ids)} roof samples")

# ============================================================
# STEP 4: TRAIN / VAL / TEST SPLIT
# ============================================================

total = len(roof_ids)

train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_ids = roof_ids[:train_end]
val_ids = roof_ids[train_end:val_end]
test_ids = roof_ids[val_end:]

print(f"Train: {len(train_ids)}")
print(f"Val:   {len(val_ids)}")
print(f"Test:  {len(test_ids)}")

# ============================================================
# STEP 5: CREATE LOOKUP TABLE
# ============================================================

file_lookup = {}

for file_path in all_files:

    stem = Path(file_path).stem

    if stem not in file_lookup:
        file_lookup[stem] = []

    file_lookup[stem].append(file_path)

# ============================================================
# STEP 6: ORGANIZE FILES
# ============================================================

def organize_split(roof_list, split_dir):

    for roof_id in roof_list:

        roof_folder = os.path.join(split_dir, roof_id)

        os.makedirs(roof_folder, exist_ok=True)

        if roof_id in file_lookup:

            for src_path in file_lookup[roof_id]:

                filename = os.path.basename(src_path)

                dst_path = os.path.join(
                    roof_folder,
                    filename
                )

                shutil.copy2(src_path, dst_path)

# ============================================================
# STEP 7: RUN ORGANIZATION
# ============================================================

print("\\nOrganizing TRAIN...")
organize_split(train_ids, train_dir)

print("Organizing VAL...")
organize_split(val_ids, val_dir)

print("Organizing TEST...")
organize_split(test_ids, test_dir)

# ============================================================
# DONE
# ============================================================

print("\\nDONE!")

