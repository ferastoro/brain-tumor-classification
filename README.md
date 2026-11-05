# Brain Tumor MRI Classification — Research-Driven Deep Learning with PyTorch

This repository contains the complete, reproducible research pipeline and Jupyter Notebook for classifying Brain Tumor MRI scans into four pathological categories: **Glioma, Meningioma, Pituitary Adenoma, and Normal (No Tumor)**.

The project rigorously compares two distinct deep learning approaches:
1. **Custom Deep CNN (Trained from Scratch):** A 4-stage convolutional neural network (~422k parameters) with Batch Normalization, Max Pooling, Global Average Pooling, and Dropout.
2. **Transfer Learning via EfficientNet-B0 (Pretrained on ImageNet):** An inverted bottleneck compound-scaled architecture (~4.01M parameters) with Squeeze-and-Excitation attention, fine-tuned across a two-phase protocol.

---

## 1. Dataset Requirements & Kaggle Source

- **Dataset Name:** Brain Tumor MRI Dataset
- **Author:** Masoud Nickparvar
- **Kaggle URL:** [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Total Images:** 7,200 MRI slices (5,600 official training, 1,600 official testing).
- **Target Classes:** 4 classes (`glioma`, `meningioma`, `notumor`, `pituitary`).

---

## 2. Running on Kaggle Notebook (Step-by-Step)

### Step 1: Create a New Notebook
1. Go to [Kaggle](https://www.kaggle.com/) and navigate to **Code** $\rightarrow$ **New Notebook**.
2. Upload `brain_tumor_classification_pytorch.ipynb` via **File** $\rightarrow$ **Upload Notebook**.

### Step 2: Attach the Dataset via "Add Input"
1. In the right-hand panel of your Kaggle notebook, click the **+ Add Input** button (or **Data** $\rightarrow$ **Add Data**).
2. In the search modal, search for:
   ```text
   masoudnickparvar/brain-tumor-mri-dataset
   ```
3. Click the **+** (Add) icon next to *Brain Tumor MRI Dataset*.
4. Verify that the dataset appears under `/kaggle/input/brain-tumor-mri-dataset` or `/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset`. The notebook dynamically auto-detects both paths.

### Step 3: Enable GPU Acceleration
1. In the right-hand sidebar under **Notebook Settings**:
   - Locate **Accelerator**.
   - Change from **None** to **GPU T4 x2** (or **GPU P100**).
2. Ensure **Internet** is toggled **ON** (required to download ImageNet pretrained weights for EfficientNet-B0).
3. Ensure **Language** is set to **Python** and **Environment** is set to **Always use latest environment**.

### Step 4: Run the Notebook
Execute cells sequentially from top to bottom (**Run** $\rightarrow$ **Run All**). Total execution time on a Kaggle T4 GPU is approximately **6 to 9 minutes**.

---

## 3. Running Locally (macOS / Linux / Windows)

The notebook automatically detects local execution environments and resolves the dataset if placed at `./dataset`:

```bash
# Clone or navigate to the workspace
cd "/Users/admin/Developer/coding_backup/SEM 5/DL/tugas-dl-1"

# Verify Python dependencies
pip install torch torchvision numpy pandas matplotlib scipy scikit-learn pillow

# Launch JupyterLab or Notebook
jupyter notebook brain_tumor_classification_pytorch.ipynb
```

Hardware acceleration automatically falls back to:
- **CUDA** if an NVIDIA GPU is present.
- **Apple Silicon MPS (`mps`)** on modern MacBooks (M1/M2/M3/M4).
- **CPU** if no accelerator is available.

---

## 4. Notebook Structure & Execution Order

The notebook `brain_tumor_classification_pytorch.ipynb` contains 35 self-contained cells organized into 16 structured research sections:

| Section | Title | Key Operations |
| :--- | :--- | :--- |
| **Section 1** | Introduction & Objectives | Clinical background, theoretical framework, research questions (RQ1–RQ4). |
| **Section 2** | Environment Setup | Reproducibility (`seed=42`), CUDA/MPS detection, Automatic Mixed Precision (AMP). |
| **Section 3** | Dataset Loading & Path Resolution | Dynamic path discovery, file scanning, corruption verification, false extension detection. |
| **Section 4** | Exploratory Data Analysis | 4-panel EDA plot, class balance, resolution distributions, 20-sample visual grid. |
| **Section 5** | Data Leakage & Group Partitioning | Exact (MD5) & near-duplicate (pHash) detection, label conflict quarantine, stratified group-aware 70/15/15 split. |
| **Section 6** | Data Preprocessing & Augmentation | PyTorch `BrainMRIDataset`, RGB channel unification, safe medical augmentations (no vertical flips). |
| **Section 7** | DataLoader Construction | Batch size 32, pin memory, multi-worker streaming, tensor shape validation. |
| **Section 8** | Model 1: Custom Deep CNN | 4 Conv blocks + BatchNorm + MaxPool + GAP + Dropout (~422k parameters). |
| **Section 9** | Model 2: EfficientNet-B0 | Replaced 4-class classification head, ImageNet weights (~4.01M parameters). |
| **Section 10** | Unified Training Pipeline | Mixed precision engine, Label Smoothing, AdamW, Cosine Annealing, Early Stopping. |
| **Section 11** | Model Evaluation | Clean test set metrics, training loss/accuracy curves, normalized confusion matrices. |
| **Section 12** | Model Comparison | Structured results table, parameter counts, latency, leakage quantification. |
| **Section 13** | Error Analysis | Top failure cases visualization, confidence inspection, diagnostic error patterns. |
| **Section 14** | Model Interpretability (Grad-CAM) | Native PyTorch Grad-CAM implementation targeting EfficientNet-B0's final conv layer. |
| **Section 15** | Research Discussion & Limitations | Synthesis answering RQ1–RQ4, patient-level leakage disclosure, academic citations. |
| **Section 16** | Deliverables Serialization | Verification and export of checkpoints, plots, and CSV reports to `/kaggle/working/`. |

---

## 5. Expected Output Deliverables

Upon completion, all deliverables are automatically saved to `/kaggle/working/` (or `./working` locally):

```
working/
├── best_custom_cnn.pth             # Optimal Custom CNN weights (highest Val Macro-F1)
├── best_efficientnet_b0.pth        # Optimal EfficientNet-B0 weights
├── model_comparison.csv            # Final comparison table across all diagnostic metrics
├── classification_report.csv       # Class-wise precision, recall, and F1 scores
├── training_history.csv            # Per-epoch training/validation loss, accuracy, and F1
├── eda_overview.png                # 4-panel EDA figure (class balance, resolution, modes, aspect ratio)
├── dataset_samples.png             # 4x5 visual sample grid across all 4 pathologies
├── training_curves.png             # Side-by-side loss and accuracy progression curves
├── confusion_matrix_comparison.png # Normalized confusion matrices for both models
├── error_analysis_samples.png      # Representative misclassified MRI failure cases
├── gradcam_examples.png            # Saliency heatmaps and overlays across pathologies
├── experiment_summary.md           # High-level Markdown executive summary
└── experiment_config.json          # Complete JSON run configuration and hyperparameter manifest
```

---

## 6. Important Methodological Limitations

1. **Slice-Level vs. Patient-Level Independence:** The dataset author anonymized filenames sequentially without providing patient IDs or DICOM headers. Because the underlying Figshare cohort contains an average of ~13 slices per patient, adjacent slices from the same subject remain dispersed across splits. **This evaluation is slice-stratified, not patient-independent.**
2. **Synthetic Pre-Augmentation Contamination:** The original Kaggle test split contains 103 pre-augmented images (`Te-aug-me_*.jpg`). Our notebook explicitly filters these out from the primary clean evaluation set.
3. **Severe Cross-Split Data Leakage in Kaggle's Default Split:** Our audit detected that **23.12% of images in Kaggle's official test directory have exact perceptual duplicates in the training directory**. Evaluating on Kaggle's default split inflates accuracy by 2 to 4 percentage points. Our notebook reports both clean and contaminated benchmarks to scientifically quantify this distortion.
4. **Shortcut Learning on Normal Controls (`notumor`):** While tumor scans are natively 512×512, normal control scans have a median resolution of 228×243 and exhibit a 32.1% border text annotation rate. Models risk learning background resolution rather than brain tissue features.
5. **Grad-CAM Interpretability Caveat:** Grad-CAM heatmaps are low-resolution visual explanations ($7 \times 7$ upsampled to $224 \times 224$) and must **never** be interpreted as clinical segmentation masks or proof of pathological understanding.

# brain-tumor-classification
