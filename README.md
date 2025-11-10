# 🧬 IDC Breast Cancer Detection using Hybrid VGG16 + InceptionV3 (Binary Classification)

This project implements a **Hybrid Transfer Learning Model** combining **VGG16** and **InceptionV3** for accurate **Invasive Ductal Carcinoma (IDC)** detection from histopathology images.  
The hybrid model integrates **feature-level fusion**, **fine-tuning**, and **advanced preprocessing** to achieve superior recall and generalization over the baseline 5-layer CNN.

---

## 📘 Overview

| Component | Description |
|------------|-------------|
| **Task** | Binary classification — IDC-positive (1) vs IDC-negative (0) |
| **Dataset** | IDC Regular PS299 patches (299×299 RGB images) |
| **Approach** | Hybrid Transfer Learning (VGG16 + InceptionV3 feature fusion) |
| **Challenge** | Class imbalance and high intraclass variance |
| **Goal** | Maximize recall while maintaining high accuracy and ROC-AUC |

---

## ⚖️ Dataset Details

| Class | Meaning | Count | Ratio |
|--------|----------|--------|--------|
| **Class 0** | Normal (non-IDC) | ~198k | 75% |
| **Class 1** | IDC-positive (malignant) | ~78k | 25% |

**Imbalance Handling:**
- On-the-fly augmentation for minority expansion  
- Weighted batch sampling for balance  
- Binary Crossentropy loss for calibrated probabilities  

---

## 🧪 Image Preprocessing

| Step | Method | Purpose |
|------|---------|----------|
| **1. CLAHE** | clipLimit≈1.0, tileGridSize=(8,8) | Enhances nuclei contrast |
| **2. Soft Circular Blend** | feather=10–12 px, softness≈0.9 | Reduces patch borders |
| **3. Color Balance Boost** | Mild boost in pink/purple tones | Clarifies nuclei and tissue structures |

All preprocessed images saved to:  
`/content/final_preprocessed`

---

## 🔄 On-the-Fly Augmentation

| Augmentation | Range | Purpose |
|---------------|--------|----------|
| **RandomFlip** | Horizontal + Vertical | Rotation invariance |
| **RandomRotation** | ±8% (~±30°) | Simulates microscope orientation |
| **RandomZoom** | ±8% | Mimics magnification changes |

🧠 *New transformations applied every epoch for infinite diversity.*

---

## 🧠 Model Architecture — Hybrid VGG16 + InceptionV3

| Component | Description |
|------------|-------------|
| **Backbones** | Pretrained VGG16 + InceptionV3 (ImageNet) |
| **Fusion** | Feature-level concatenation (GlobalAvgPool outputs) |
| **Head** | Dense(1024) → BN → Dropout(0.55) ×2 → Dense(1, Sigmoid) |
| **Regularization** | L2 (1e-4) + Strong Dropout |
| **Optimizer** | Adam (1e-4 → 1e-5) |
| **Loss** | Binary Crossentropy |
| **Input** | 299×299×3 |
| **Batch Size** | 32 |
| **Epochs** | 5 (frozen) + 8 (fine-tuned) |

---

## 📈 Training Summary

| Stage | Description | Train Acc | Val Acc | Val Loss |
|--------|--------------|------------|-----------|-----------|
| **Stage 1** | Dense head only (frozen backbones) | 0.8729 | 0.8774 | 0.3056 |
| **Stage 2** | Fine-tuned top layers | 0.9090 | **0.9071** | 0.2263 |

✅ **Final Model:** `/content/hybrid_vgg_inception_final_binary.h5`  
✅ **Best Checkpoint:** `/content/hybrid_vgg_inc_best_binary.h5`

---

## 📊 Evaluation Metrics

| Metric | Threshold = 0.5 | Optimal (Youden’s J ≈ 0.3899) |          
|:---------|:---------------:|:-----------------------------:|
| **Accuracy** | 0.9109 | 0.9039 |
| **Precision** | 0.8310 | 0.7912 |
| **Recall (IDC)** | 0.8615 | **0.8986** |
| **F1-score** | 0.8460 | 0.8415 |
| **ROC-AUC** | **0.9643** | **0.9643** |   


## 🧾 Classification Reports

### Classification Report @ Threshold = 0.5
| Metric | Precision | Recall | F1-Score | Support |
|:--------|:-----------:|:--------:|:----------:|:----------:|
| **Class 0 (Normal)** | 0.9443 | 0.9305 | 0.9374 | 19875 |
| **Class 1 (IDC)** | 0.8310 | 0.8615 | 0.8460 | 7880 |
| **Accuracy** |  |  | **0.9109** | 27755 |
| **Macro Avg** | 0.8876 | 0.8960 | 0.8917 |  |
| **Weighted Avg** | 0.9121 | 0.9109 | 0.9114 |  |

---

### Classification Report @ Optimal Threshold (≈ 0.3899)
| Metric | Precision | Recall | F1-Score | Support |
|:--------|:-----------:|:--------:|:----------:|:----------:|
| **Class 0 (Normal)** | 0.9575 | 0.9060 | 0.9310 | 19875 |
| **Class 1 (IDC)** | 0.7912 | 0.8986 | 0.8415 | 7880 |
| **Accuracy** |  |  | **0.9039** | 27755 |
| **Macro Avg** | 0.8743 | 0.9023 | 0.8862 |  |
| **Weighted Avg** | 0.9103 | 0.9039 | 0.9056 |  |

---

📊 **Insight:**  
At the optimized threshold, the model achieves **higher recall (0.8986)** for IDC cases while maintaining strong accuracy and precision — crucial for minimizing false negatives in cancer detection.

## 🧮 Confusion Matrices

| Threshold | TN | FP | FN | TP |
|------------|----|----|----|----|
| **0.5** | 18494 | 1381 | 1091 | 6789 |
| **0.3899 (opt)** | 18006 | 1869 | 799 | 7081 |

**→ False negatives reduced by ~26%.**

---

### 🔷 Visual Results

**Confusion Matrices**
<img width="1364" height="728" alt="Screenshot 2025-11-10 034915" src="https://github.com/user-attachments/assets/1a7b92d9-2656-443c-8dd5-4986bfe10ece" />
**AUC-ROC ** 
<img width="723" height="651" alt="Screenshot 2025-11-10 034850" src="https://github.com/user-attachments/assets/4a948510-d1fd-41fc-a831-6581f1bb0199" />


---

## 🧩 Model Comparison — Base CNN vs Hybrid Model

| Metric | 🧱 5-Layer CNN | 🚀 Hybrid VGG16 + InceptionV3 |
|:--------|:---------------:|:------------------------------:|
| **Accuracy** | 86.14 % | **91.09 %** |
| **ROC-AUC** | 0.91 | **0.964** |
| **Recall (IDC)** | 0.84 | **0.8986** |
| **Loss** | 0.328 | **0.226** |
| **False Negatives** | High | **Reduced by ~26 %** |

✅ The hybrid model achieves **higher accuracy, AUC, and recall**, making it more reliable for diagnostic use.

---


**Numbers used:**  
- Base 5-layer CNN — Accuracy: 86.14%, Recall: 84.00%, ROC-AUC: 0.9100  
- Hybrid VGG16 + InceptionV3 — Accuracy: 91.09%, Recall: 89.86%, ROC-AUC: 96.43  

---

## 🏆 Key Highlights

- Processed **histopathology images** using CLAHE enhancement and soft circular blending.  
- Applied **real-time geometric augmentations** (flip, rotation, zoom) to improve robustness.  
- Fine-tuned a **Hybrid VGG16 + InceptionV3** model achieving **91.09% accuracy**, **0.964 ROC-AUC**, and **0.8986 recall (IDC)**.  
- Reduced false negatives by **~26%**, improving medical reliability and diagnostic safety.  

---

## 📂 Repository Structure


