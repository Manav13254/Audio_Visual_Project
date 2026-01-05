# Multimodal Audio–Visual Aerial Scene Classification

This project investigates **multimodal learning for aerial scene recognition** by jointly leveraging **RGB aerial imagery and environmental audio**. A **cross-attention–based fusion model** is proposed and evaluated on the ADVANCE dataset.

---

## Overview
- Task: Aerial scene classification (13 classes)
- Modalities: Vision + Audio
- Dataset: ADVANCE (5,075 paired samples)
- Objective: Compare unimodal baselines against multimodal fusion

---

## Method
- **Vision encoder:** CLIP RN50 (frozen) with CBAM attention
- **Audio encoder:** ResNet-18 (frozen) with SENet attention
- **Fusion:** Cross-Attention Block (CAB) enabling bidirectional audio–visual refinement

---

## Key Results
- Best unimodal (vision): **91.46% accuracy**
- Best unimodal (audio): **79.00% accuracy**
- **Multimodal CAB fusion:** **93.62% accuracy**

Multimodal fusion consistently outperforms unimodal models.

---

## Interpretability
Grad-CAM analysis shows improved spatial focus and more semantically meaningful activations in the multimodal model compared to unimodal baselines.

---

## Authors
- Manav Jobanputra  
- Pranav Khunt  

Supervisor: Dr. Ankit Jha
