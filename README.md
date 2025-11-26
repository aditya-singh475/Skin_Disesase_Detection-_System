# Skin_Disesase_Detection_System

# 🧠  Skin Disease Detection System

An intelligent skin disease detection system powered by deep learning. This project classifies skin conditions such as **Melanoma**, **Benign Mole**, **Acne**, **Fungal Infection** using image inputs. It also provides **confidence scores**, **Grad-CAM heatmaps**, and **interactive visualizations** to enhance transparency and trust.

---

## 📌 Problem Statement

- 1 in 3 people globally suffer from skin diseases.
- Late detection of melanoma and infections can be life-threatening.
- Dermatologists are not accessible to all, and diagnosis may vary.
- AI offers fast, consistent, and scalable diagnosis support.

---

## 🎯 Objectives

- Classify skin conditions into 5 categories using CNN-based models.
- Display prediction confidence and Grad-CAM heatmaps.
- Build a responsive, user-friendly UI using Streamlit.
- Promote early awareness through educational visuals.

---

## 🧪 Methodology

```mermaid
graph TD;
    A[Upload Image] --> B[Lesion Detection (YOLOv8)];
    B --> C[Disease Classification (ResNet/EfficientNet)];
    C --> D[Confidence Score + Grad-CAM Heatmap];
    D --> E[Interactive UI Output];
##  Dataset
Melanoma & Benign Mole: ISIC / HAM10000

Acne & Fungal: Kaggle dermatology datasets

Split: Train (70%) / Validation (15%) / Test (15%)

Diversity: Includes varied skin tones, lighting, and angles

Augmentation: Used to balance limited disease samples

