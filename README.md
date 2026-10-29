# Thermal-Target-Detector: Counter-UAS Sky-Facing Classifier

An end-to-end computer vision pipeline designed for sky-facing Counter-UAS (Unmanned Aerial Systems) threat detection. The system classifies low-RCS aerial targets (commercial drones) against biological clutter (birds) using a custom PyTorch binary Convolutional Neural Network (CNN) trained on simulated Thermal Infrared (TIR) sensor streams.

## Key Technical Features
* **Simulated Thermal Sensor Preprocessing:** Ingests sky-facing optical feeds, converts them to single-channel grayscale to replicate 8-bit uncooled TIR hardware, and applies **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to maximize silhouette and edge contrast against cloud and sky gradients.
* **Custom Data Pipeline:** Automated PyTorch `Dataset` loader that parses directory structures and maps filename conventions to binary classes (`1` for target drones, `0` for background birds).
* **Real-Time Visual Inference:** Evaluates batch streams via OpenCV with dynamic confidence scoring and color-coded status overlays (`TARGET DRONE` vs. `BACKGROUND BIRD`).

## Project Structure
* `config.py` - Global configuration parameters (image size, learning rate, batch size).
* `data_loader.py` - Single-channel grayscale conversion and CLAHE edge-enhancement pipeline.
* `model.py` - Binary CNN architecture optimized for low-footprint classification.
* `train.py` - Training script featuring automated weight serialization (`thermal_cnn_weights.pth`).
* `test_visual.py` - Interactive OpenCV batch evaluation and visual overlay tester.

## Quickstart
1. Place training data in the `data/` directory (prefix target drone files with `target_`).
2. Train the model:
   ```bash
   python train.py