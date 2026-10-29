import os

# CLAHE Preprocessing Parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

# Model Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
IMAGE_SIZE = (128, 128) # Adjust to match your input dimensions if different

# Detection Confidence Threshold
CONFIDENCE_THRESHOLD = 0.85

# File Paths
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "thermal_cnn_weights.pth")