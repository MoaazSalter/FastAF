# Model Training

This folder contains the training configuration, results, and metrics for all four YOLOv11 models developed during the FastAF drug detection system. Each iteration improved on the previous through better data, tuned hyperparameters, and architectural decisions.

The final production model is **Model 4**.

---

## Dataset Overview

All models were trained on a custom drug detection dataset built and annotated specifically for this project. The dataset is hosted on Roboflow and covers **46 drug classes** across common Egyptian pharmacy products.

**Dataset:** [FastAF Drug Detection Dataset on Roboflow](https://universe.roboflow.com/test-c1hxb/fastaf/dataset/4)

The dataset was collected and built through two sources:

- **1,868 photos** captured manually using 3 different phones across varied real-world conditions: different lighting, backgrounds, angles, and positions. Some images intentionally include damaged packaging, cluttered shelves, and unrelated objects in the background.
- **1,041 photos** scraped from the web using SerpAPI to supplement underrepresented classes.

All images were manually annotated using Roboflow.

> **Note on data growth across versions:** The 1,868 manually captured photos represent the total accumulated across all collection rounds and are fully present in the final dataset (v4). Each earlier version contains a subset of this — v1 and v3 were built from earlier, smaller collection rounds, with more photos added in each subsequent version. The dataset grew incrementally: every new version is a superset of the manually captured data from the version before it.

### Why These 46 Drugs

The drug classes were deliberately chosen to challenge the model with hard cases:

- **Near-identical packaging, different products** — 4 Voltaren variants (Topical Gel, 75mg Ampoule, SR 100mg Tablet, 100mg Suppository), 3 Ceftriaxone variants (1000mg I.M., 1000mg I.V., 500mg I.M.), 3 Concor variants, 2 Otrivin variants, 2 DG Wash variants, 3 Telfast variants
- **Non-box flexible packaging** — Dove Topical Cream, Seropipe Hair Serum, Limitless Woman/Man/Omega-3 bottles
- **Multiple SKUs per class** — Sensodyne Fluoride (20ml/50ml/100ml tubes) are all treated as a single class; the model detects drug identity and the database handles SKU-level pricing

### Dataset Versions

| Version | Used In | Training Images | Validation Images | Notes |
|---------|---------|-----------------|-------------------|-------|
| v1 | Model 1 | 3,345 | 278 | Roboflow augmentation applied (flip, rotation, salt & pepper noise) producing 3× source images |
| v3 | Model 2 | 1,445 | 254 | Clean dataset, no Roboflow augmentation — runtime augmentation used during training instead |
| v4 | Models 3 & 4 | 7,065 | 442 | Largest dataset (7,507 total images); Roboflow augmentation: rotation, brightness, Gaussian blur |

> **Note on validation sets:** Each dataset version has its own independent validation split. The 278 images used to evaluate Model 1 are not the same images as the 254 used for Model 2, and neither overlaps with the 442 used for Models 3 and 4. All splits are **stratified** — each class is proportionally represented in both train and validation sets. This means metric comparisons across models reflect genuine performance differences and are not an artifact of easier or harder validation sets.

---

## Model Comparison

| | Model 1 | Model 2 | Model 3 | Model 4 ✅ |
|---|---|---|---|---|
| **Architecture** | YOLOv11m | YOLOv11m | YOLOv11s | YOLOv11m |
| **Parameters** | 20M | 20M | 9.4M | 20M |
| **Optimizer** | SGD (auto) | SGD (auto) | SGD (auto) | AdamW |
| **Learning rate** | 0.01 | 0.01 | 0.01 | 0.0003 |
| **Dataset** | v1 | v3 | v4 | v4 |
| **Training images** | 3,345 | 1,445 | 7,065 | 7,065 |
| **Validation images** | 278 | 254 | 442 | 442 |
| **Epochs trained** | 60 | 150 | 198 | 214 |
| **mAP50** | 0.942 | 0.963 | 0.977 | **0.980** |
| **mAP50-95** | 0.830 | 0.829 | 0.852 | **0.856** |
| **Precision** | 0.904 | 0.920 | **0.958** | 0.955 |
| **Recall** | 0.904 | 0.922 | 0.945 | **0.965** |
| **Inference (prod)** | ~11ms | ~11ms | **~5ms** | ~11ms |
| **Weight file size** | 40.6 MB | 40.6 MB | **19.2 MB** | 40.6 MB |

> Inference times measured on Tesla T4 GPU.

---

## Model 1 — Baseline

**Folder:** `model_1_yolo11m_datasetv1`

The first trained model, establishing the baseline. Used dataset v1 which was heavily augmented by Roboflow to produce 3× the source images (horizontal flip, vertical flip, 90° rotations, ±15° rotation, salt & pepper noise). Trained for 60 epochs with standard SGD and default YOLOv11 hyperparameters.

**Key observations:**
- Achieved 0.942 mAP50 in only 60 epochs, a strong baseline given the dataset quality
- Main weakness: `Voltaren 100mg Rectal Suppositories` scored only 0.560 mAP50
- `Fucidin Topical Cream` recall was only 0.375 — frequently missed
- `Signal medium Toothpaste` mAP50 of 0.737, struggling against similar Signal variants

---

## Model 2 — Better Augmentation and More Epochs

**Folder:** `model_2_yolo11m_datasetv3`

Same YOLOv11m architecture but switched to dataset v3 (clean, no Roboflow augmentation) and moved augmentation to runtime training instead. Added rotation (±20°), vertical flip, mixup (10%), and copy-paste (10%) augmentation. Trained for 150 epochs with early stopping patience of 15.

**Key observations:**
- mAP50 improved from 0.942 → 0.963 (+2.1%)
- Fixed model 1's worst class: `Voltaren 100mg Rectal Suppositories` jumped from 0.560 → 0.919 mAP50
- `Fucidin Topical Cream` went from 0.375 recall to 1.0 recall — completely resolved
- mAP50-95 stayed essentially flat at 0.829 (vs 0.830), showing tight-box precision was already saturated at this data scale

---

## Model 3 — Bigger Dataset

**Folder:** `model_3_yolo11s_datasetv4`

switching to the **small** YOLOv11s architecture (9.4M params, half the size of medium) on the much larger dataset v4 (7,065 training images) outperformed both previous medium models on every metric. Early stopping triggered at epoch 198 (best at epoch 180).

**Key observations:**
- mAP50: 0.977, mAP50-95: 0.852 — best so far on both metrics
- Inference speed doubled vs medium models (5.4ms vs ~11ms) — directly beneficial for real-time WebSocket streaming
- Model weight only 19.2MB vs 40.6MB
- Proved that data scale matters more than model capacity at this problem size
- `DG wash fluoride` was the only clear weak class (mAP50-95: 0.732), likely due to only 9 validation images

---

## Model 4 — Production Model ✅

**Folder:** `model_4_yolo11m_datasetv4`

The final production model. Returned to the medium architecture (YOLOv11m) to pair it with the large v4 dataset, and replaced SGD with **AdamW** optimizer (lr=0.0003) — a well-suited optimizer for transformer-hybrid architectures. Extended `close_mosaic` from 10 to 70 epochs so mosaic augmentation ran through more of the critical mid-training phase. Trained up to 214 epochs.

**Why Model 4 over Model 3:**
- Improved mAP50: 0.977 → **0.980**
- Improved mAP50-95: 0.852 → **0.856**
- Improved Recall: 0.945 → **0.965** — the model misses fewer drugs, critical for a sales environment
- The medium architecture has more capacity to distinguish visually similar drug classes at high confidence; since inference runs on a cloud server (not a mobile device), the 2× speed advantage of the small model is not a constraint

**AdamW + low lr impact:** The lower learning rate (0.0003 vs 0.01) combined with AdamW's weight decay decoupling produced a smoother convergence curve. The model reached the 0.978 mAP50 range by epoch ~100 and continued refining through epoch 214 with very small but consistent gains.

---

## Training Environment

All models were trained on Google Colab using a **Tesla T4 GPU (15GB VRAM)** with:
- Python 3.11
- PyTorch 2.6.0 + CUDA 12.4
- Ultralytics 8.3.x
- Batch size: 16
- Image size: 640×640

