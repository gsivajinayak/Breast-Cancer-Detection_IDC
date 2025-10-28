# 🧬 IDC Breast Cancer Detection using 5-Layer CNN (Imbalanced Binary Classification)

This project implements a **custom 5-layer CNN** for **Invasive Ductal Carcinoma (IDC)** detection from histopathology images.  
The model is trained **from scratch (no fine-tuning)** on an **imbalanced binary dataset** with **research-style preprocessing** and **on-the-fly geometric augmentation** to ensure strong generalization.

---

## 📘 Overview

| Component | Description |
|------------|-------------|
| **Task** | Binary classification — IDC-positive (1) vs IDC-negative (0) |
| **Dataset** | IDC Regular PS50 patches (50×50 RGB images) |
| **Challenge** | Highly imbalanced dataset (IDC minority) |
| **Solution** | Preprocessing + real-time augmentation + strong dropout CNN |
| **Goal** | Accurate IDC patch detection with high recall and stable generalization |

---

## ⚖️ Dataset Imbalance

| Class | Meaning | Count | Ratio |
|--------|----------|--------|--------|
| **Class 0** | Normal (non-IDC) | ~198k | 75% |
| **Class 1** | IDC-positive (malignant) | ~78k | 25% |

**Imbalance Handling:**
- On-the-fly geometric augmentation (Flip, Rotate, Zoom)
- Balanced batch sampling
- Strong dropout regularization
- Binary Crossentropy loss for calibrated probability outputs

---

## 🧪 Preprocessing

Applied once and saved under `/content/final_preprocessed`.

| Step | Method | Purpose |
|------|---------|----------|
| **1. CLAHE** | clipLimit≈1.0, tileGridSize=(8,8) | Enhances local contrast without over-sharpening |
| **2. Soft Circular Blend** | feather=10–12 px, softness=0.85–0.9 | Removes patch borders, focuses on tissue |
| **3. Color Balance Boost** | Mild lift in white, pink, purple | Clarifies nuclei and stromal tissue patterns |

---

## 🔄 On-the-Fly Augmentation (Training Only)

| Augmentation | Range | Purpose |
|---------------|--------|----------|
| **RandomFlip** | Horizontal + Vertical | Orientation variability |
| **RandomRotation** | ±8% (~±30°) | Simulates slide rotation |
| **RandomZoom** | ±8% | Mimics magnification differences |

🧠 *Every epoch → new random transformations for each image → infinite diversity without saving augmented images.*

---

## 🧠 Model Architecture — 5-Layer CNN (No Fine-Tuning)

| Block | Filters | Dropout | Description |
|--------|----------|----------|--------------|
| Conv2D(32) + BN + MaxPool | 0.40 | Edge & texture extraction |
| Conv2D(64) + BN + MaxPool | 0.45 | Cell-level features |
| Conv2D(128) + BN + MaxPool | 0.50 | Glandular structure learning |
| Conv2D(256) + BN + MaxPool | 0.55 | Deep tissue patterns |
| Conv2D(256) + BN + GAP | 0.60 | Global tissue representation |
| Dense(256, ReLU) + Dropout(0.5) | — | Fully connected classifier |
| Dense(1, Sigmoid) | — | Binary prediction (IDC / Normal) |

**Training Config:**  
- Optimizer: Adam (lr=1e-4)  
- Loss: Binary Crossentropy  
- Input: 50×50×3  
- Batch Size: 128  
- Epochs: 20  

---

## 📊 Performance

| Metric | Score |
|--------|--------|
| **Accuracy** | **86.14%** |
| **Precision (IDC)** | 0.72 |
| **Recall (IDC)** | **0.84** |
| **F1-score (IDC)** | 0.77 |
| **ROC-AUC** | ,0.91 |

### Confusion Matrix
<img width="620" height="436" alt="image" src="https://github.com/user-attachments/assets/6496d8a7-8823-4b69-a32c-af436f9c793b" />
## ROC 
<img width="732" height="642" alt="image" src="https://github.com/user-attachments/assets/f97b4533-b60d-421f-b15b-cd72ddf9c0bb" />
