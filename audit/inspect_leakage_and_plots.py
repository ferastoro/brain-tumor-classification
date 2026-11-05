import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
import matplotlib.pyplot as plt

df = pd.read_csv('image_audit_metadata.csv')

# 1. Investigate the 4 PNG files and unusual modes (RGBA, P)
print("--- Unusual Format/Mode Files ---")
unusual = df[(df['format'] != 'JPEG') | (~df['mode'].isin(['RGB', 'L']))]
print(unusual[['split', 'class', 'filename', 'format', 'mode', 'width', 'height']])

# 2. Check the cross-class pHash conflict
phash_groups = df.groupby('phash')
for pval, grp in phash_groups:
    if len(grp['class'].unique()) > 1:
        print("\n--- Cross-Class pHash Conflict ---")
        print(grp[['split', 'class', 'filename', 'width', 'height', 'phash']])

# 3. Check aug-me vs me
print("\n--- Meningioma Subtypes Analysis ---")
me_df = df[df['class'] == 'meningioma']
me_df['prefix'] = me_df['filename'].apply(lambda x: x.split('_')[0])
print(me_df.groupby(['split', 'prefix']).size())

# 4. Generate Class Distribution Bar Chart
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
splits = ['Training', 'Testing']
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
x = np.arange(len(classes))
width = 0.35

train_counts = [len(df[(df['split']=='Training') & (df['class']==c)]) for c in classes]
test_counts = [len(df[(df['split']=='Testing') & (df['class']==c)]) for c in classes]

rects1 = ax.bar(x - width/2, train_counts, width, label='Training Split (N=5,600)', color='#1f77b4', edgecolor='black')
rects2 = ax.bar(x + width/2, test_counts, width, label='Testing Split (N=1,600)', color='#aec7e8', edgecolor='black')

ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
ax.set_title('Class Distribution Across Official Training & Testing Splits\n(Brain Tumor MRI Dataset)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels([c.capitalize() for c in classes], fontsize=11, fontweight='bold')
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_ylim(0, 1650)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('class_distribution.png')
plt.close()
print("Saved class_distribution.png")

# 5. Generate dataset_samples.png
# Sample 5 images from each of the 4 classes (from training & testing)
np.random.seed(42)
sampled_images = []
for c in classes:
    class_df = df[df['class'] == c]
    # pick 3 from training, 2 from testing
    train_sample = class_df[class_df['split'] == 'Training'].sample(3, random_state=42)
    test_sample = class_df[class_df['split'] == 'Testing'].sample(2, random_state=42)
    sample_df = pd.concat([train_sample, test_sample])
    sampled_images.append(sample_df)

sample_all = pd.concat(sampled_images).reset_index(drop=True)

fig, axes = plt.subplots(4, 5, figsize=(15, 12), dpi=200)
for row_idx, c in enumerate(classes):
    sub = sample_all[sample_all['class'] == c].reset_index(drop=True)
    for col_idx in range(5):
        ax = axes[row_idx, col_idx]
        item = sub.iloc[col_idx]
        img = Image.open(item['path'])
        ax.imshow(img, cmap='gray' if img.mode == 'L' else None)
        ax.axis('off')
        ax.set_title(f"{c.capitalize()} ({item['split'][:2]})\n{item['width']}x{item['height']} | {item['mode']}\n{item['filename']}", fontsize=8)

plt.suptitle("Representative Samples from Brain Tumor MRI Dataset\n(5 images per class, showing resolution, mode, split)", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('dataset_samples.png')
plt.close()
print("Saved dataset_samples.png")

# 6. Generate Cross-Split Leakage Pair Visualization
# Let's find pairs with pHash distance 0 between Training and Testing
train_df = df[df['split'] == 'Training']
test_df = df[df['split'] == 'Testing']

phash_matches = []
for pval, grp in phash_groups:
    splits = grp['split'].unique()
    if 'Training' in splits and 'Testing' in splits:
        tr_items = grp[grp['split'] == 'Training']
        te_items = grp[grp['split'] == 'Testing']
        for _, tr_row in tr_items.iterrows():
            for _, te_row in te_items.iterrows():
                phash_matches.append((tr_row, te_row))

print(f"Total exact pHash match pairs (Train vs Test): {len(phash_matches)}")

# Plot 4 sample leakage pairs (one for each class if available)
fig, axes = plt.subplots(4, 2, figsize=(8, 14), dpi=200)
classes_represented = []
leak_samples = []
for tr, te in phash_matches:
    if tr['class'] not in classes_represented:
        classes_represented.append(tr['class'])
        leak_samples.append((tr, te))
    if len(classes_represented) == 4:
        break

for i, (tr, te) in enumerate(leak_samples):
    # Train
    ax_tr = axes[i, 0]
    img_tr = Image.open(tr['path'])
    ax_tr.imshow(img_tr, cmap='gray' if img_tr.mode == 'L' else None)
    ax_tr.axis('off')
    ax_tr.set_title(f"TRAIN: {tr['class'].capitalize()}\n{tr['filename']} ({tr['width']}x{tr['height']})", fontsize=9, fontweight='bold', color='darkblue')

    # Test
    ax_te = axes[i, 1]
    img_te = Image.open(te['path'])
    ax_te.imshow(img_te, cmap='gray' if img_te.mode == 'L' else None)
    ax_te.axis('off')
    ax_te.set_title(f"TEST: {te['class'].capitalize()}\n{te['filename']} ({te['width']}x{te['height']})", fontsize=9, fontweight='bold', color='darkred')

plt.suptitle("EVIDENCE OF CROSS-SPLIT DATA LEAKAGE\nIdentical Perceptual Hashes (pHash Distance = 0) Between Train & Test", fontsize=12, fontweight='bold', color='darkred', y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('cross_split_leakage_samples.png')
plt.close()
print("Saved cross_split_leakage_samples.png")
