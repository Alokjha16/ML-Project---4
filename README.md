# 🚀 DINOv2 Semantic Segmentation Model

## 📌 Overview
This repository contains a semantic segmentation model built using the DINOv2 Vision Transformer as a backbone and a convolutional decoder for dense prediction tasks.

The model predicts 10 semantic categories at pixel level resolution.

---

## 🏗️ Model Configuration

- **Backbone:** DINOv2 (dinov2_vits14)
- **Decoder:** CNN segmentation head
- **Input Resolution:** 252 × 448
- **Classes:** 10
- **Framework:** PyTorch

---

## 💡 Core Characteristics

- Vision Transformer encoder
- Feature refinement using CNN layers
- Optimized training loop
- Clean inference pipeline

---

## 📊 Evaluation Metrics

| Metric | Score |
|--------|--------|
| IoU | **0.4442** |


---

## 🖼️ Prediction Example

<img width="1002" height="498" alt="Screenshot 2026-02-28 014613" src="https://github.com/user-attachments/assets/95ddef14-490c-4a2b-8959-e49c479174af" />


---

## ▶️ Running the Project

```bash
python train_segmentation.py
python generate_submissions.py
python evaluate_variants.py
```

---

## 🧰 Environment

- Python
- PyTorch
- OpenCV
- NumPy
