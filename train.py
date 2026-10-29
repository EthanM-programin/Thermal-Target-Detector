import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import config
from model import ThermalCNN
from data_loader import ThermalDataset

def train_model():
    """Executes the full training pipeline for the Thermal Target Detector CNN,
    tracking loss curves and evaluating performance across configured epochs.
    """
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"[INFO] Initializing Thermal-Target-Detector Training Pipeline")
    print(f"[INFO] Target Device: {device}")
    print(f"[INFO] Batch Size: {config.BATCH_SIZE} | Learning Rate: {config.LEARNING_RATE}")
    print("=" * 60)

    # Define standard image augmentations and tensor conversions
    data_transforms = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Locate dataset path
    dataset_path = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(dataset_path):
        print(f"[WARNING] Data directory not found at: {dataset_path}")
        print("[INFO] Please ensure your training images are placed in a 'data/' folder.")
        return

    # Instantiate dataset and dataloader
    dataset = ThermalDataset(image_dir=dataset_path, transform=data_transforms)
    train_loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    print(f"[INFO] Dataset loaded successfully. Total samples: {len(dataset)}")

    # Initialize model, loss criterion, and optimizer
    model = ThermalCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Training loop across epochs
    model.train()
    print("\n[INFO] Starting model optimization loop...")

    for epoch in range(config.NUM_EPOCHS):
        epoch_start = time.time()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero gradients, forward pass, compute loss, backpropagate, and step optimizer
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = (correct_predictions / total_samples) * 100.0
        epoch_duration = time.time() - epoch_start

        # Print structured epoch progress
        print(f"Epoch [{epoch + 1:02d}/{config.NUM_EPOCHS:02d}] | "
              f"Loss: {epoch_loss:.4f} | "
              f"Accuracy: {epoch_acc:6.2f}% | "
              f"Duration: {epoch_duration:.2f}s")

    # Save trained model weights to configured destination
    os.makedirs(os.path.dirname(config.WEIGHTS_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.WEIGHTS_PATH)

    total_duration = time.time() - start_time
    print("=" * 60)
    print(f"[SUCCESS] Training complete in {total_duration:.2f} seconds.")
    print(f"[SUCCESS] Optimized model weights saved to: {config.WEIGHTS_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    train_model()