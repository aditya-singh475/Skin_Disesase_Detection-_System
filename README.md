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
## Methodlogy
graph TD;
A[Upload Image] --> B[Lesion Detection (YOLOv8)];
B --> C[Disease Classification (EfficientNet)];
C --> D[Image Enhancement];
D --> E[Interactive GUI Output];



Augmentation: Used to balance limited disease samples

