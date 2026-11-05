import os
import hashlib
import json
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

dataset_path = '/Users/admin/Developer/coding_backup/SEM 5/DL/tugas-dl-1/dataset'

records = []
corrupted = []

print("Scanning all images...")
for split in ['Training', 'Testing']:
    for cls in ['glioma', 'meningioma', 'notumor', 'pituitary']:
        dirpath = os.path.join(dataset_path, split, cls)
        files = sorted([f for f in os.listdir(dirpath) if not f.startswith('.')])
        for fname in files:
            fpath = os.path.join(dirpath, fname)
            size = os.path.getsize(fpath)
            try:
                with open(fpath, 'rb') as f:
                    content = f.read()
                    md5 = hashlib.md5(content).hexdigest()
                    sha256 = hashlib.sha256(content).hexdigest()
                with Image.open(fpath) as img:
                    w, h = img.size
                    mode = img.mode
                    fmt = img.format
                    phash_val = imagehash.phash(img)
                    dhash_val = imagehash.dhash(img)
                records.append({
                    'split': split,
                    'class': cls,
                    'filename': fname,
                    'path': fpath,
                    'size_bytes': size,
                    'width': w,
                    'height': h,
                    'mode': mode,
                    'format': fmt,
                    'md5': md5,
                    'sha256': sha256,
                    'phash': str(phash_val),
                    'dhash': str(dhash_val),
                    'phash_obj': phash_val,
                    'dhash_obj': dhash_val
                })
            except Exception as e:
                corrupted.append((fpath, str(e)))

df = pd.DataFrame(records)
print(f"Total images audited: {len(df)}")
print(f"Total corrupted: {len(corrupted)}")

# Save basic metadata to CSV for reference
df_save = df.drop(columns=['phash_obj', 'dhash_obj'])
df_save.to_csv('image_audit_metadata.csv', index=False)

# 1. Total dataset size
total_bytes = df['size_bytes'].sum()
print(f"Total dataset size: {total_bytes / (1024*1024):.2f} MB ({total_bytes / (1024*1024*1024):.4f} GB)")

# 2. Formats and Modes
print("\n--- Image Formats ---")
print(df['format'].value_counts())
print("\n--- Image Modes (Channels) ---")
print(df['mode'].value_counts())

# Check if 3 channels actually have RGB identical channels (grayscale encoded as RGB)
print("\nChecking if RGB images are actually grayscale (R == G == B)...")
sample_indices = np.random.choice(len(df), size=min(500, len(df)), replace=False)
is_actually_gray = 0
for idx in sample_indices:
    row = df.iloc[idx]
    with Image.open(row['path']) as img:
        arr = np.array(img)
        if len(arr.shape) == 3 and arr.shape[2] == 3:
            if np.array_equal(arr[:,:,0], arr[:,:,1]) and np.array_equal(arr[:,:,1], arr[:,:,2]):
                is_actually_gray += 1
print(f"Sample of 500 images: {is_actually_gray}/500 ({is_actually_gray/5:.1f}%) have identical R=G=B channels.")

# 3. Dimension Distribution
print("\n--- Dimensions Summary ---")
print(f"Width: min={df['width'].min()}, max={df['width'].max()}, mean={df['width'].mean():.1f}, median={df['width'].median():.1f}")
print(f"Height: min={df['height'].min()}, max={df['height'].max()}, mean={df['height'].mean():.1f}, median={df['height'].median():.1f}")
print("Top 10 most frequent resolutions:")
print(df.groupby(['width', 'height']).size().sort_values(ascending=False).head(10))

# 4. Class Distribution
print("\n--- Class Distribution by Split ---")
dist = pd.crosstab(df['class'], df['split'], margins=True)
print(dist)

# 5. Exact Duplicates (MD5)
print("\n--- Exact Duplicate Analysis (MD5) ---")
md5_groups = df.groupby('md5')
exact_dup_groups = {k: v for k, v in md5_groups if len(v) > 1}
num_exact_dup_images = sum(len(v) for v in exact_dup_groups.values())
num_unique_hashes = len(md5_groups)
print(f"Unique MD5 hashes: {num_unique_hashes} out of {len(df)} total files")
print(f"Number of duplicate groups: {len(exact_dup_groups)}")
print(f"Total images involved in exact duplicates: {num_exact_dup_images} (Redundant files: {num_exact_dup_images - len(exact_dup_groups)})")

# Check split and class distribution of exact duplicates
cross_split_exact = 0
cross_class_exact = 0
within_train_exact = 0
within_test_exact = 0

cross_split_exact_cases = []
cross_class_exact_cases = []

for md5_hash, group in exact_dup_groups.items():
    splits = set(group['split'])
    classes = set(group['class'])
    if len(splits) > 1:
        cross_split_exact += 1
        cross_split_exact_cases.append(group[['split', 'class', 'filename']].to_dict(orient='records'))
    elif 'Training' in splits:
        within_train_exact += 1
    elif 'Testing' in splits:
        within_test_exact += 1
    
    if len(classes) > 1:
        cross_class_exact += 1
        cross_class_exact_cases.append(group[['split', 'class', 'filename']].to_dict(orient='records'))

print(f"Exact duplicates crossing Train/Test split: {cross_split_exact} groups")
print(f"Exact duplicates strictly within Training: {within_train_exact} groups")
print(f"Exact duplicates strictly within Testing: {within_test_exact} groups")
print(f"Exact duplicates across DIFFERENT classes (label conflict!): {cross_class_exact} groups")

# 6. Near Duplicate Analysis (pHash)
print("\n--- Perceptual Hash (pHash) Analysis ---")
phash_groups = df.groupby('phash')
phash_exact_groups = {k: v for k, v in phash_groups if len(v) > 1}
num_phash_dup_images = sum(len(v) for v in phash_exact_groups.values())
print(f"Images with exact identical pHash (Hamming distance 0): {num_phash_dup_images} in {len(phash_exact_groups)} groups")

cross_split_phash0 = 0
cross_class_phash0 = 0
for phash_val, group in phash_exact_groups.items():
    if len(set(group['split'])) > 1:
        cross_split_phash0 += 1
    if len(set(group['class'])) > 1:
        cross_class_phash0 += 1
print(f"pHash dist=0 crossing Train/Test split: {cross_split_phash0} groups")
print(f"pHash dist=0 across DIFFERENT classes: {cross_class_phash0} groups")

# Calculate pairwise pHash distance for small Hamming distances (e.g. <= 2) across splits
# To make it efficient, we can compare hashes using bitwise operations
hashes = [row['phash_obj'] for _, row in df.iterrows()]
splits = list(df['split'])
classes = list(df['class'])
fnames = list(df['filename'])

train_idx = [i for i, s in enumerate(splits) if s == 'Training']
test_idx = [i for i, s in enumerate(splits) if s == 'Testing']

print(f"\nChecking pairwise near-duplicates (Hamming dist <= 2) between Train ({len(train_idx)}) and Test ({len(test_idx)})...")
# Convert hashes to uint64 array for ultra-fast vectorized hamming distance
hash_ints = np.array([int(str(h), 16) for h in hashes], dtype=np.uint64)

# Function to compute popcount on numpy uint64
def hamming_distance_matrix(arr1, arr2):
    # arr1: N, arr2: M
    xor_mat = arr1[:, None] ^ arr2[None, :]
    # bitcount
    # In python 3.10+, int.bit_count() exists. For numpy, bitwise popcount:
    # Use bitwise tricks or python list comprehension in chunks
    return xor_mat

near_cross_leakage_dist0 = 0
near_cross_leakage_dist1_2 = 0
leakage_examples = []

train_hash_ints = hash_ints[train_idx]
test_hash_ints = hash_ints[test_idx]

# Check pairwise efficiently
# Compare each test image against all train images
for t_local, t_global in enumerate(test_idx):
    t_h = hashes[t_global]
    # quick filter: check dist against all train hashes
    # To keep it very fast:
    diffs = [t_h - hashes[tr_global] for tr_global in train_idx]
    for d, tr_global in zip(diffs, train_idx):
        if d == 0:
            near_cross_leakage_dist0 += 1
            if len(leakage_examples) < 10:
                leakage_examples.append({
                    'train_file': fnames[tr_global],
                    'train_class': classes[tr_global],
                    'test_file': fnames[t_global],
                    'test_class': classes[t_global],
                    'distance': d
                })
        elif d <= 2:
            near_cross_leakage_dist1_2 += 1
            if len(leakage_examples) < 10:
                leakage_examples.append({
                    'train_file': fnames[tr_global],
                    'train_class': classes[tr_global],
                    'test_file': fnames[t_global],
                    'test_class': classes[t_global],
                    'distance': d
                })

print(f"Cross-split identical pHash pairs (dist=0): {near_cross_leakage_dist0}")
print(f"Cross-split near-identical pHash pairs (dist 1-2): {near_cross_leakage_dist1_2}")

# Save detailed report dict
summary = {
    'total_images': len(df),
    'total_bytes': int(total_bytes),
    'formats': df['format'].value_counts().to_dict(),
    'modes': df['mode'].value_counts().to_dict(),
    'exact_dup_groups': len(exact_dup_groups),
    'exact_dup_total_images': num_exact_dup_images,
    'cross_split_exact_groups': cross_split_exact,
    'cross_class_exact_groups': cross_class_exact,
    'within_train_exact_groups': within_train_exact,
    'within_test_exact_groups': within_test_exact,
    'cross_split_phash0_pairs': near_cross_leakage_dist0,
    'cross_split_phash1_2_pairs': near_cross_leakage_dist1_2,
    'leakage_examples': leakage_examples[:10]
}

with open('audit_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nAudit pipeline analysis complete. Summary saved to audit_summary.json")
