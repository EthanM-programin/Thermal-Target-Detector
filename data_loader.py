import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ThermalDataset(Dataset):
    """
    Custom PyTorch Dataset that generates synthetic thermal sensor frames,
    applies CLAHE constrast enhancement, and prepares tensors for model training.
    """
    def __init__(self, num_samples=1000, img_size=(128, 128)):
        self.num_samples = num_samples
        self.img_size = img_size

        # Create CLAHE object: Clip Limit controls contrast, TileGridSize sets local grid divisions
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # Randomly assign target classes: 0 = Drone, 1 = Bird
        self.labels = np.random.randint(0, 2, size=num_samples)

    def __len__(self):
        return self.num_samples

    def _generate_synthetic_thermal_frame(self, label):
        # 1. Generate noisy grey thermal background (simulating cold sky/clouds)
        bg = np.random.normal(loc=80, scale=15, size=self.img_size).astype(np.uint8)

        # 2. Add target signature near the center
        center = (self.img_size[1] // 2 + np.random.randint(-15, 15),
                  self.img_size[0] // 2 + np.random.randint(-15, 15))

        if label == 0: # Drone: Angular/box shape with rotors (brighter thermal hot spots)
            cv2.rectangle(bg, (center[0]-10, center[1]-5), (center[0]+10, center[1]+5), 220, -1)
            cv2.circle(bg, (center[0]-12, center[1]-5), 4, 255, -1)
            cv2.circle(bg, (center[0]+12, center[1]-5), 4, 255, -1)
        else: # Bird: Curved oval shape with wing angle signature
            cv2.ellipse(bg, center, (12, 5), 30, 0, 360, 180, -1)

        # 3. Add Gaussian blur to simulate atmospheric noise and sensor distance
        blurred = cv2.GaussianBlur(bg, (5, 5), 0)
        return blurred

    def __getitem__(self, idx):
        label = self.labels[idx]
        raw_frame = self._generate_synthetic_thermal_frame(label)

        # 4. Apply CLAHE enhancement to make subtle target edges pop
        enhanced_frame = self.clahe.apply(raw_frame)

        # 5. Normalize pixel value from [0, 255] to [0.0, 1.0] and add channel dimension
        tensor_frame = torch.tensor(enhanced_frame, dtype=torch.float32).unsqueeze(0) / 255.0

        return tensor_frame, torch.tensor(label, dtype=torch.long)

def get_data_loaders(batch_size=32, num_samples=1000):
    """
    Helper function to instantiate and split dataset into Train and Validation loaders.
    """
    dataset = ThermalDataset(num_samples=num_samples)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

if __name__ == "__main__":
    # Test execution
    train_loader, val_loader = get_data_loaders()
    images, labels = next(iter(train_loader))
    print(f"Data Loader Test Successful.")
    print(f"Batch Image Tensor Shape: {images.shape} (Batch Size, Channels, Height, Width)")
    print(f"Batch Label Tensor Shape: {labels.shape}")