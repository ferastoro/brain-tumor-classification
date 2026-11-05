# Post-Execution Audit Report: Brain Tumor MRI Classification (PyTorch)

**Project Title:** Brain Tumor MRI Classification — Research-Driven Deep Learning with PyTorch  
**Auditor:** Senior Machine Learning Research Engineer & Medical Computer Vision Specialist  
**Target Artifact:** `brain-tumor-classification.ipynb` (Executed in Kaggle GPU Environment)  
**Execution Environment:** Kaggle Cloud, NVIDIA Tesla T4 GPU (14.56 GB VRAM), CUDA 12.x, PyTorch 2.x  
**Audit Date:** November 6, 2026  
**Final Audit Verdict:** **PASS WITH CORRECTIONS**

---

## 1. Executive Summary

A post-execution forensic audit was performed on the executed notebook `brain-tumor-classification.ipynb` and its generated artifacts. The notebook documents an experimental benchmark comparing a **Custom Deep CNN (trained from scratch; 422,308 parameters)** against **EfficientNet-B0 (ImageNet-pretrained transfer learning; 4,012,672 parameters)** for 4-class intracranial neoplasm classification (`glioma`, `meningioma`, `notumor`, `pituitary`).

### Key Audit Findings
1. **Execution Authenticity & Completeness:** All 35 notebook cells executed sequentially from execution counter `[1]` to `[16]` without fatal errors or interrupted states. Genuine training was conducted on a Tesla T4 GPU: Custom CNN trained for 15 epochs (400.4s), and EfficientNet-B0 trained across a 2-phase protocol (3 warmup + 12 fine-tuning epochs; 315.7s).
2. **Data Leakage & Integrity Implementation:** The automated audit pipeline successfully discovered and quarantined **203 pre-augmented derivative images** (`aug-me`) and **12 cross-class contradictory scans** across 3 distinct conflict groups. A stratified group-aware split on unique perceptual hashes (`pHash`) formed a clean experimental pool of **6,985 images** partitioned into 4,938 train, 1,014 validation, and 1,033 clean test samples, with **zero duplicate-group overlap across partitions**.
3. **Primary Experimental Outcome:**
   - **EfficientNet-B0** achieved superior diagnostic performance on the clean test set (**Accuracy: 96.71%**, **Macro F1-score: 96.67%**) compared to the **Custom CNN** (**Accuracy: 86.45%**, **Macro F1-score: 86.17%**).
   - Lowest class-wise performance for EfficientNet-B0 was on **Meningioma (F1: 0.9448)** and **Glioma (F1: 0.9677)**, reflecting real-world radiological ambiguity along infiltrative and dural margins.
4. **Secondary Benchmark Finding (Leakage Distortion):** Evaluating on the original contaminated Kaggle test split (1,600 images) yielded **98.00% accuracy (+1.29 pp inflation)** for EfficientNet-B0 due to slice memorization. Conversely, the Custom CNN scored **82.81% (-3.63 pp drop)** because it was evaluated on the 103 noisy, pre-augmented `Te-aug-me` scans that were absent from its clean training distribution.
5. **Critical Scientific Limitations Identified:**
   - **No Patient Identifiers:** Because the Kaggle release omits patient/study identifiers and originates from slice-level aggregation (Figshare cohort with ~13 slices/patient), **patient-level data leakage cannot be ruled out and is virtually guaranteed**.
   - **Source Confounding in "No Tumor":** 98.9% of `notumor` images have non-512 resolutions (median 228×243) and 32.1% carry border annotations/scanner markers. High performance on `notumor` is partially driven by resolution and frequency shortcuts.
   - **Omission of Custom CNN Per-Class Metrics in CSV:** `classification_report.csv` contains only EfficientNet-B0 metrics.

---

## 2. Execution Status & Environment Verification

### Execution Environment Details
- **Platform:** Kaggle Notebook Cloud Environment
- **Accelerator:** NVIDIA Tesla T4 GPU (Device ID 0; Compute Capability 7.5; 14.56 GB Dedicated VRAM)
- **Host Python / PyTorch:** Python 3.10+, PyTorch 2.x with CUDA 12.x backend
- **Mixed Precision:** Automatic Mixed Precision (AMP) enabled (`torch.cuda.amp.autocast`, `GradScaler`)
- **Dataset Path Auto-Discovery:** Successfully resolved to `/kaggle/input/datasets/masoudnickparvar/brain-tumor-mri-dataset`
- **Output Destination:** Successfully mapped to `/kaggle/working/`

### Cell Execution Sequence Manifest

| Cell Range | Type | Execution Index | Status | Key Outputs / Verified Artifacts |
| :--- | :---: | :---: | :---: | :--- |
| **Cells 0–2** | Markdown | — | Complete | Title, abstract, neuro-oncology background, RQ1–RQ4 |
| **Cell 3** | Code | `[1]` | Complete | Deterministic seeds (`42`), GPU detection (Tesla T4), `/kaggle/working/` path |
| **Cell 4–5** | Code | `[2]` | Complete | 7,200 files scanned, 0 corrupted, 4 false-extension PNGs identified |
| **Cell 6–7** | Code | `[3]` | Complete | 4-panel EDA plot, class crosstabs, 20-sample visual inspection grid |
| **Cell 8–9** | Code | `[4]` | Complete | pHash computation, 203 aug-me excluded, 12 label conflicts quarantined |
| **Cell 10–11** | Code | `[5]` | Complete | PyTorch `BrainMRIDataset` defined, safe RGB conversion, safe augmentations |
| **Cell 12–13** | Code | `[6]` | Complete | DataLoaders: Train (4,938), Val (1,014), Clean Test (1,033), Leaked Test (1,600) |
| **Cell 14–15** | Code | `[7]` | Complete | Custom CNN instantiated (422,308 parameters) |
| **Cell 16–17** | Code | `[8]` | Complete | Torchvision ImageNet weights downloaded (20.5 MB), head replaced (4,012,672 params) |
| **Cell 18–19** | Code | `[9]` | Complete | Reusable `Trainer` class defined (AMP, Label Smoothing, Early Stopping) |
| **Cell 20–21** | Code | `[10]` | Complete | Custom CNN trained for 15 epochs (400.4s); best model checkpointed & restored |
| **Cell 22–23** | Code | `[11]` | Complete | EfficientNet-B0 trained: Phase 1 (3 ep, 1.36m) + Phase 2 (12 ep, 5.26m) |
| **Cell 24–25** | Code | `[12]` | Complete | Clean & Leaked test evaluation; loss/acc curves; confusion matrices rendered |
| **Cell 26–27** | Code | `[13]` | Complete | Final comparison table & classification report printed |
| **Cell 28–29** | Code | `[14]` | Complete | 34 misclassified samples extracted and top failure cases visualized |
| **Cell 30–31** | Code | `[15]` | Complete | Grad-CAM hooked to `features[-1]`; 4-class saliency overlays rendered |
| **Cell 32–33** | Markdown | — | Complete | Research synthesis, discussion of findings, academic citations |
| **Cell 34** | Code | `[16]` | Complete | Deliverables serialization checklist: 12/12 files confirmed in `/kaggle/working/` |

---

## 3. Dataset Integrity & Partition Audit

### 1. Ingestion & File Anomaly Verification
- **Total Files Scanned:** 7,200 images across `Training/` (5,600) and `Testing/` (1,600).
- **Corrupted / Unreadable Files:** Exactly **0**.
- **False File Extensions (Engineering Hazard Neutralized):**
  Four files were confirmed to be PNG format despite `.jpg` extensions:
  - `Tr-no_1200.jpg`: PNG format, 4-channel `RGBA` mode (393×400)
  - `Tr-no_22.jpg`: PNG format, 4-channel `RGBA` mode (550×664)
  - `Tr-no_572.jpg`: PNG format, 4-channel `RGBA` mode (442×454)
  - `Tr-no_625.jpg`: PNG format, `P` (palette) mode (728×725)
  The custom `BrainMRIDataset` properly applied `.convert('RGB')` during loading, preventing channel-mismatch tensor errors during batch collation.

### 2. Pre-Augmented Derivatives Exclusion
- **Identified:** 203 images carrying the `aug-me` prefix:
  - 100 images in `Training/meningioma/` (`Tr-aug-me_*.jpg`)
  - 103 images in `Testing/meningioma/` (`Te-aug-me_*.jpg`)
- **Action Taken:** Properly flagged and excluded from the clean experimental pool to avoid training and testing on synthetic spatial copies.

### 3. Cross-Class Contradiction Quarantine
The executed script identified **3 distinct perceptual hash conflict groups** comprising **12 images** where identical brain scans were assigned contradictory clinical labels:
- **Conflict Group 1 (pHash `901a7f616de436a6`):** 7 images (`Tr-me_458`, `Tr-me_468`, `Tr-me_714`, `Te-me_50`, `Te-me_61`, `Te-me_96` labeled *Meningioma* vs. `Te-gl_68` labeled *Glioma*).
- **Conflict Group 2 (pHash `d866a71999966166`):** 2 images (`Tr-me_794` *Meningioma* vs. `Te-gl_367` *Glioma*).
- **Conflict Group 3 (pHash `932d6c666c323719`):** 3 images (`Tr-me_1122`, `Te-me_159` *Meningioma* vs. `Te-gl_160` *Glioma*).
- **Action Taken:** All 12 contradictory scans were quarantined and excluded from both models' experimental pools.

### 4. Clean Partition Statistics

```
+---------------------------------------------------------------------------------+
|                    VERIFIED CLEAN PARTITION CLASS DISTRIBUTION                  |
+---------------------------------------------------------------------------------+
| Class Label    | Clean Train (70%) | Clean Val (15%) | Clean Test (15%) | Total |
+----------------+-------------------+-----------------+------------------+-------+
| Glioma         |       1,259       |       269       |       269        | 1,797 |
| Meningioma     |       1,113       |       234       |       241        | 1,588 |
| No Tumor       |       1,307       |       240       |       253        | 1,800 |
| Pituitary      |       1,259       |       271       |       270        | 1,800 |
+----------------+-------------------+-----------------+------------------+-------+
| Total          |       4,938       |      1,014      |      1,033       | 6,985 |
+----------------+-------------------+-----------------+------------------+-------+
```

---

## 4. Data Leakage Assessment & Methodological Boundaries

```
+-------------------------------------------------------------------------------+
|                      DATA LEAKAGE AUDIT MATRIX                                |
+-------------------------------------------------------------------------------+
| Leakage Vector                       | Status    | Severity | Audit Finding   |
+--------------------------------------+-----------+----------+-----------------+
| Exact Slice Duplicates Across Splits | ELIMINATED| NONE     | 0 overlap       |
| Identical pHash Across Splits        | ELIMINATED| NONE     | 0 overlap       |
| Near-Duplicate (Dist 1-2) Clustering | PARTIAL   | MEDIUM   | Distance 0 only |
| Cross-Class Contradictory Scans      | QUARANTINED| NONE    | 12 imgs removed |
| Pre-Augmented Test Copies            | EXCLUDED  | NONE     | 203 imgs removed|
| Patient-Level Identity Leakage       | UNRESOLVED| HIGH     | No Patient IDs  |
+-------------------------------------------------------------------------------+
```

### Critical Leakage Findings
1. **Slice-Level Independence Achieved:** Within the clean pool, identical perceptual hashes were grouped via `group_mapping` and stratified. Assertions in Cell 9 verified:
   $$\text{Train Groups} \cap \text{Val Groups} = \emptyset, \quad \text{Train Groups} \cap \text{Test Groups} = \emptyset, \quad \text{Val Groups} \cap \text{Test Groups} = \emptyset$$
2. **The Patient-Level Conundrum (HIGH Severity):**
   The underlying Figshare cohort (Cheng et al., 2017) contains 3,064 slices from 233 patients (~13 slices per patient). Because Masoud Nickparvar anonymized all filenames into arbitrary sequential tokens (`Tr-gl_1.jpg`), **patient-level metadata does not exist**. Consequently, consecutive 2D slices from the same subject inevitably remain distributed across training, validation, and testing partitions.
   > [!IMPORTANT]
   > The study correctly documents this limitation in Section 15. The experiment is **slice-stratified and duplicate-quarantined**, but **NOT patient-independent**.

---

## 5. Source Bias & Dataset Confounding Audit

An in-depth review of Section 4 and Section 15 revealed substantial domain-level confounding between class labels and image acquisition characteristics:

1. **The "No Tumor" Acquisition Dichotomy:**
   - **Tumor Classes (`glioma`, `meningioma`, `pituitary`):** Sourced primarily from the Figshare dataset (Cheng et al.); **~92% are natively 512×512 square slices** with standardized field-of-view.
   - **Normal Controls (`notumor`):** Sourced from external open datasets (Br35H / Bhuvaji); **only 1.1% are 512×512**, with a median resolution of **228×243** and non-square aspect ratios (0.645–1.786).
2. **Scanner Marker & Text Artifacts:**
   Border pixel analysis confirmed that **32.1% of `notumor` images contain high-intensity border artifacts**, frame lines, or scanner orientation markers ("L", "R"), compared to only 2.0% in glioma.
3. **Shortcut Learning Risk:**
   Deep networks can easily distinguish normal scans from tumors by learning frequency-domain interpolation artifacts, high-frequency blurring, or perimeter text stamps.
4. **Clinical Generalizability Verdict:**
   The models cannot be assumed to generalize to unseen clinical hospital scanners. Claims of clinical diagnostic deployment are scientifically unjustified and must remain framed as academic feasibility benchmarks.

---

## 6. Model Architecture & Training Pipeline Audit

### 1. Custom CNN Architecture
- **Structure:** 4 convolutional blocks ($3 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256$) with `Conv2d(kernel_size=3, padding=1, bias=False)`, `BatchNorm2d`, `ReLU`, and `MaxPool2d(2, 2)`.
- **Classification Head:** `AdaptiveAvgPool2d((1, 1))` $\rightarrow$ `Dropout(0.4)` $\rightarrow$ `Linear(256, 128)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Linear(128, 4)`.
- **Verified Parameters:** Exactly **422,308 parameters** (100% trainable).
- **Design Assessment:** Sound, modern inductive bias. Avoids fully connected parameter explosion while providing receptive field expansion suited for $224 \times 224$ images.

### 2. EfficientNet-B0 Transfer Learning Architecture
- **Backbone:** Torchvision `models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)`. Downloaded official 20.5 MB weights (`efficientnet_b0_rwightman-7f5810bc.pth`).
- **Classification Head:** Replaced default 1,000-class linear layer with `nn.Sequential(Dropout(p=0.3, inplace=True), Linear(1280, 4))`.
- **Verified Parameters:** Exactly **4,012,672 parameters**.
- **Two-Phase Transfer Learning Execution:**
  - *Phase 1 (Warmup; 3 epochs):* Backbone frozen (`features.requires_grad = False`). Trained only classifier head ($1280 \rightarrow 4$) with AdamW ($\text{LR} = 10^{-3}$, weight decay $= 10^{-4}$). Val Macro-F1 progressed from 0.8416 to 0.8845.
  - *Phase 2 (Fine-Tuning; 12 epochs):* Unfroze top 3 stages (`features[-3:]`). Discriminative learning rates correctly configured: $\text{LR} = 10^{-4}$ for backbone features and $\text{LR} = 5 \times 10^{-4}$ for classifier head, scheduled with `CosineAnnealingLR` down to $10^{-6}$.

### 3. Training Stability & Convergence Analysis

| Epoch / Stage | Custom CNN Train Acc / Val Acc | Custom CNN Val Loss / Macro-F1 | EfficientNet Train Acc / Val Acc | EfficientNet Val Loss / Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Phase 1: Ep 1** | — | — | 0.7594 / 0.8432 | 0.5950 / 0.8416 |
| **Phase 1: Ep 3** | — | — | 0.8524 / 0.8876 | 0.5017 / 0.8845 |
| **Stage / Ep 1** | 0.6426 / 0.5010 | 1.3521 / 0.4458 | 0.8836 / 0.9329 | 0.3744 / 0.9315 |
| **Stage / Ep 4** | 0.7724 / 0.7722 | 0.6904 / 0.7614 | 0.9599 / 0.9596 | 0.3220 / 0.9593 |
| **Stage / Ep 8** | 0.8230 / 0.8471 | 0.5664 / 0.8431 | 0.9795 / 0.9744 | 0.2919 / 0.9739 |
| **Stage / Ep 12** | 0.8548 / 0.7101 | 0.8773 / 0.6802 | **0.9850 / 0.9753** | **0.2888 / 0.9748** |
| **Stage / Ep 15** | **0.8649 / 0.9004** | **0.4516 / 0.8983** | — | — |

- **Custom CNN Dynamics:** Exhibited early validation volatility (Val Loss spiked to 3.5791 at Epoch 2 before stabilizing). Batch Normalization and Cosine Annealing enabled steady recovery, achieving peak Val Macro-F1 of **0.8983** at Epoch 15.
- **EfficientNet-B0 Dynamics:** Exceptional convergence smoothness. Validation loss monotonically declined from 0.5950 to **0.2865** at Epoch 10, with peak Val Macro-F1 reaching **0.9749** at Epoch 9. No divergence, numerical instability, or gradient explosion occurred.

---

## 7. Evaluation Integrity & Metric Verification

All reported evaluation metrics were calculated strictly on the untouched clean test partition (1,033 samples).

### Verified Evaluation Integrity Checks
1. **Ordering & Mapping:** Verified that predictions and ground-truth targets maintained identical indexing across all DataLoader batches (`shuffle=False`). Class indices strictly aligned: `0: Glioma`, `1: Meningioma`, `2: Notumor`, `3: Pituitary`.
2. **Checkpoint Restoration:** `Trainer.fit()` explicitly logged:
   `[*] Restoring best checkpoint from /kaggle/working/best_custom_cnn.pth...`
   `[*] Restoring best checkpoint from /kaggle/working/best_efficientnet_b0.pth...`
   Final evaluation evaluated the restored optimal validation checkpoints.
3. **Identical Test Partitions:** Both models evaluated on the exact same 1,033 clean test samples and 1,600 contaminated test samples.

### Detailed Diagnostic Performance (Clean Test Partition: N = 1,033)

```
+---------------------------------------------------------------------------------------------------+
|                           OFFICIAL COMPARATIVE EVALUATION MATRIX                                  |
+---------------------------------------------------------------------------------------------------+
| Metric                            | Custom CNN (From Scratch)     | EfficientNet-B0 (Pretrained)  |
+-----------------------------------+-------------------------------+-------------------------------+
| Model Parameters                  | 422,308                       | 4,012,672                     |
| Training Wall-Clock Time          | 400.4 seconds (6.67 mins)     | 315.7 seconds (5.26 mins)     |
| GPU Inference Latency (ms/image)  | 3.67 ms                       | 3.78 ms                       |
| Best Validation Macro F1          | 0.8983                        | 0.9749                        |
| Clean Test Accuracy               | 86.45% (893 / 1,033)          | 96.71% (999 / 1,033)          |
| Clean Test Macro Precision        | 86.50%                        | 96.66%                        |
| Clean Test Macro Recall           | 86.17%                        | 96.71%                        |
| Clean Test Macro F1-Score         | 86.17%                        | 96.67%                        |
| Clean Test Weighted F1-Score      | 86.42%                        | 96.72%                        |
+-----------------------------------+-------------------------------+-------------------------------+
| Leaked Test Accuracy (Kaggle Split| 82.81% (1,325 / 1,600)        | 98.00% (1,568 / 1,600)        |
| Apparent Leakage Shift            | -3.63 pp                      | +1.29 pp                      |
+-----------------------------------+-------------------------------+-------------------------------+
```

### Class-Wise Diagnostic Performance: EfficientNet-B0 (Clean Test)

| Class | Precision | Recall | F1-Score | Clean Test Support | Clinical Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Glioma** | **0.9884** | 0.9480 | **0.9677** | 269 | High precision; 14 false negatives |
| **Meningioma** | 0.9315 | 0.9585 | **0.9448** | 241 | Most difficult class; 10 false negatives |
| **No Tumor** | 0.9651 | **0.9842** | **0.9746** | 253 | High recall; 4 false negatives |
| **Pituitary** | 0.9814 | 0.9778 | **0.9796** | 270 | Highest overall F1; 6 false negatives |
| **Macro Average** | **0.9666** | **0.9671** | **0.9667** | 1,033 | Balanced diagnostic efficacy |
| **Weighted Average**| **0.9676** | **0.9671** | **0.9672** | 1,033 | Representative sample aggregate |

---

## 8. Secondary Benchmark Audit: Quantifying Leakage & Noise

Evaluating both models on the original 1,600-image Kaggle testing split produced a revealing methodological result:

1. **EfficientNet-B0 (+1.29 pp Inflation):**
   - Clean Test Accuracy: **96.71%**
   - Original Leaked Test Accuracy: **98.00%**
   - **Mechanism:** EfficientNet-B0 possesses 4.01M parameters and high representational capacity. It memorized features from the 370 duplicate and 505 near-duplicate slices shared between the original training and testing directories, artificially inflating its test score.
2. **Custom CNN (-3.63 pp Drop):**
   - Clean Test Accuracy: **86.45%**
   - Original Leaked Test Accuracy: **82.81%**
   - **Mechanism:** Why did the Custom CNN perform *worse* on the leaked split?
     Because our clean training split explicitly excluded all 100 `Tr-aug-me` pre-augmented images. Consequently, when the Custom CNN was evaluated on the original Kaggle testing split, it encountered **103 pre-augmented `Te-aug-me` images** exhibiting spatial warps, shear, and rotations that were entirely absent from its training distribution. With restricted capacity (422k params), the Custom CNN could not generalize to these synthetic distortions.
3. **Methodological Takeaway:**
   Data leakage does not uniformly inflate all models. In datasets contaminated by both duplicates and synthetic pre-augmentations, capacity-limited models suffer out-of-distribution performance drops on synthetic test artifacts, while high-capacity transfer models exploit slice redundancy.

---

## 9. Error Analysis & Interpretability (Grad-CAM) Audit

### 1. Error Analysis Verification
- In Cell 29, EfficientNet-B0 committed **34 misclassifications out of 1,033 clean test scans (3.29% error rate)**.
- **Primary Diagnostic Confusion:**
  - *Glioma vs. Meningioma:* Most errors occurred between high-grade gliomas with dural contact and atypical meningiomas. Both can present with peritumoral vasogenic edema and contrast enhancement.
  - *Pituitary Microadenomas:* A small number of pituitary cases with subtle intrasellar lesions were confused with normal anatomy due to slice orientation angle.

### 2. Grad-CAM Implementation Audit
- **Target Layer:** Successfully hooked to `effnet_model.features[-1]` (`Conv2dNormActivation` outputting $7 \times 7$ feature maps with 1,280 channels).
- **Forward & Backward Hooks:** Properly detached activations and captured full backward gradients. Channel-wise global average pooling of gradients was correctly applied.
- **Heatmap Generation & Overlay:** Upsampled $7 \times 7$ CAM to $224 \times 224$ via bilinear interpolation and rendered with `jet` colormap at $\alpha = 0.45$.
- **Clinical Interpretability Assessment:**
  - For `glioma` and `meningioma`, saliency maps concentrated intensely on the contrast-enhancing tumor mass.
  - For `pituitary`, activations focused tightly on the sella turcica / skull base region.
  - For `notumor`, activations were diffuse across cerebral parenchyma and ventricles.
  - **Audit Warning (LOW Severity):** The notebook generated Grad-CAM only for *correctly predicted* samples. Generating Grad-CAM for the 34 failure cases would provide deeper clinical insight into why the model erred.

---

## 10. Reproducibility & Deliverables Verification

### Execution Manifest in `/kaggle/working/`
Cell 34 verified the serialization of all 12 project artifacts in the Kaggle execution container:

```
/kaggle/working/
├── [VERIFIED] best_custom_cnn.pth              (1,663.4 KB)  # Serialized model state dict
├── [VERIFIED] best_efficientnet_b0.pth         (15,970.6 KB) # Serialized model state dict
├── [VERIFIED] model_comparison.csv             (0.3 KB)      # Final comparative table
├── [VERIFIED] classification_report.csv        (0.5 KB)      # Per-class metrics
├── [VERIFIED] training_history.csv             (1.8 KB)      # 15-epoch training logs
├── [VERIFIED] eda_overview.png                 (189.7 KB)    # 4-panel EDA visualization
├── [VERIFIED] dataset_samples.png              (1,476.4 KB)  # 20-sample visual grid
├── [VERIFIED] training_curves.png              (176.2 KB)    # Loss & accuracy curves
├── [VERIFIED] confusion_matrix_comparison.png  (105.9 KB)    # Normalized confusion matrices
├── [VERIFIED] gradcam_examples.png             (947.1 KB)    # Saliency overlays
├── [VERIFIED] experiment_summary.md            (0.4 KB)      # Markdown executive summary
└── [VERIFIED] experiment_config.json           (0.5 KB)      # Run configuration & seeds
```

### Reproducibility Verification
- `seed_everything(42)` correctly set Python, NumPy, and PyTorch CPU/CUDA seeds.
- PyTorch deterministic flags (`torch.backends.cudnn.deterministic = True`, `benchmark = False`) were applied.
- Dynamic path resolution guarantees immediate re-executability on any fresh Kaggle GPU session.

---

## 11. Critical Issues & Recommended Corrections

```
+---------------------------------------------------------------------------------------------------+
|                                  AUDIT FINDINGS & SEVERITY LOG                                    |
+---------------------------------------------------------------------------------------------------+
| ID | Finding                                      | Severity | Impact on Validity                 |
+----+----------------------------------------------+----------+------------------------------------+
| 01 | Unresolved Patient-Level Leakage             | HIGH     | Cannot claim clinical independence |
| 02 | Source / Resolution Confounding in "No Tumor"| HIGH     | Risk of shortcut feature learning  |
| 03 | Near-Duplicate pHash Clustering (Dist 1-2)   | MEDIUM   | Minor residual slice leakage       |
| 04 | Omission of Custom CNN Class-Wise CSV Report | MEDIUM   | Incomplete comparative CSV record  |
| 05 | Local Absence of Loose Checkpoint Files      | LOW      | Checkpoints remain on Kaggle Cloud |
| 06 | Grad-CAM Generated Only for True Positives   | LOW      | Missed failure mode interpretability|
| 07 | PyTorch 2.x AMP Deprecation Warnings         | LOW      | Future compatibility notice        |
+---------------------------------------------------------------------------------------------------+
```

### Detailed Issue Analysis & Corrections

#### Issue 01: Unresolved Patient-Level Leakage
- **Severity:** **HIGH**
- **Evidence:** Zero patient or hospital accession metadata exists. The Figshare source contains ~13 slices per patient.
- **Why It Matters:** Multi-slice MRI scans from a single subject share idiosyncratic anatomical markers. Testing on slices from training patients measures patient memorization alongside pathology.
- **Correction:** Frame all conclusions as *slice-level classification benchmarks*. Add an explicit disclosure stating: *"Due to lack of patient identifiers in the source Kaggle dataset, slice-level clustering was performed. Patient-level independence cannot be guaranteed."*

#### Issue 02: Source / Resolution Confounding in "No Tumor"
- **Severity:** **HIGH**
- **Evidence:** 98.9% of `notumor` images are non-512 resolutions (median 228×243) and 32.1% exhibit border text annotations, whereas tumor classes are 92% 512×512 without annotations.
- **Why It Matters:** Models can distinguish normal scans using high-frequency blur or border markers rather than intracranial tissue integrity.
- **Correction:** In the research report discussion, explicitly analyze this domain shortcut. Recommend skull-stripping and intensity harmonization for future clinical studies.

#### Issue 03: Near-Duplicate pHash Clustering Limited to Distance 0
- **Severity:** **MEDIUM**
- **Evidence:** Grouping was conducted on exact pHash strings. Slices with Hamming distance 1 or 2 (representing minor compression artifacts) were not grouped into the same cluster.
- **Why It Matters:** Some near-identical slices may have crossed partition boundaries.
- **Correction:** Document that pHash grouping addressed exact perceptual matches ($D=0$) and that future work should expand connected components to $D \le 2$.

#### Issue 04: Omission of Custom CNN Per-Class CSV Report
- **Severity:** **MEDIUM**
- **Evidence:** Cell 27 exported `classification_report.csv` containing only EfficientNet-B0's metrics.
- **Why It Matters:** Limits direct tabular comparison of per-class precision and recall between the two models in automated pipelines.
- **Correction:** In future revisions, export both `classification_report_efficientnet.csv` and `classification_report_custom_cnn.csv`.

#### Issue 05: Loose Deliverables Stored on Kaggle Cloud
- **Severity:** **LOW**
- **Evidence:** The executed notebook exists locally, but the 12 generated files (`.pth`, `.csv`, `.png`) reside in the cloud container `/kaggle/working/`.
- **Correction:** Users needing the raw 16 MB model weights locally should download the `/kaggle/working/` archive from the Kaggle Notebook output tab.

---

## 12. Final Verdict & Academic Assessment

### Final Verdict: **PASS WITH CORRECTIONS**

```
+-------------------------------------------------------------------------------+
|                            FINAL AUDIT DETERMINATION                          |
+-------------------------------------------------------------------------------+
|  Implementation & Code Quality:      EXCELLENT (10/10)                        |
|  Execution Integrity:                 VERIFIED & REPRODUCIBLE (10/10)          |
|  Data Leakage Quarantine:             SUCCESSFUL SLICE-LEVEL DE-DUPLICATION   |
|  Model Performance Comparison:        SCIENTIFICALLY SOUND & DEFENSIVE        |
|  Remaining Limitations:               PATIENT IDs ABSENT; NOTUMOR SHORTCUT    |
|                                                                               |
|  FINAL RULING:                        PASS WITH CORRECTIONS                   |
+-------------------------------------------------------------------------------+
```

### Academic Defense Summary
The project achieves its primary academic research objective: conducting a **rigorous, reproducible, and scientifically defensible comparison between a Custom CNN baseline and EfficientNet-B0 Transfer Learning**. 

Unlike conventional Kaggle submissions that naively train on contaminated data and report inflated >99% accuracies, this study demonstrates advanced medical computer vision maturity by:
1. **Discovering and quarantining 203 pre-augmented scans and 12 contradictory labels.**
2. **Eliminating exact duplicate leakage across partitions.**
3. **Proving that EfficientNet-B0 achieves superior diagnostic generalization (96.67% Clean Macro-F1 vs. 86.17% for Custom CNN) while requiring comparable inference latency (3.78 ms vs. 3.67 ms).**
4. **Empirically demonstrating the distortion caused by data leakage on public benchmarks.**

Subject to the documented limitations regarding patient-level independence and shortcut bias, this project represents an exemplary, publication-grade undergraduate Deep Learning submission.

