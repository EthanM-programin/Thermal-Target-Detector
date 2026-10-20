# 🛰️ Thermal Target Classification & Real-Time Surveillance Engine

A computer vision pipeline built with PyTorch and OpenCV designed to classify low-contrast aerial thermal targets (Drones vs. Birds) under simulated atmospheric noise conditions.

## 👁️ Pipeline Architecture
1. **Preprocessing (CLAHE):** Applies Contrast Limited Adaptive Histogram Equalization across local tile grids to extract faint thermal target edges from noisy infrared backgrounds.
2. **Deep Learning (Custom CNN):** A convolutional neural network trained on multi-channel feature maps to recognize structural target signatures.
3. **Real-Time Inference (`detect.py`):** Feeds live synthetic sensor streams through the trained weights, outputting bounding box coordinates and live confidence scores via OpenCV.

## 📊 Performance Metrics
* **Final Validation Accuracy:** 99.50%
* **Training Accuracy:** 97.38%

## 🚀 Execution
```bash
# Train the Convolutional Neural Network
python train.py

# Launch the real-time OpenCV detection surveillance window
python detect.py