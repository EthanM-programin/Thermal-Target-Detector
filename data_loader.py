import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ThermalDataset(Dataset):
    """
    Custom PyTorch Dataset that loads sky-facing imagery, converts them to
    single-channel grayscale to simulate uncooled Thermal Infrared (TIR) sensors,
    applies CLAHE edge-enhancement, and assigns binary classification labels.
    """
    def __init__(self, image_dir: str, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Gather all valid image files from the target directory
        valid_extensions = ('.png', '.jpg', '.jpeg')
        self.image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(valid_extensions)
        ]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # 1. Load image in single-channel grayscale (simulating 8-bit thermal sensor output)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Unable to read image stream at: {img_path}")

        # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This sharpens low-RCS targets and thermal gradients against cloud/sky clutter
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        # 3. Convert processed OpenCV numpy array to a PIL Image format
        img_pil = Image.fromarray(img)

        # 4. Apply torchvision pipeline transforms (Resize, ToTensor, Normalize)
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)

        # 5. Automated Class Labeling Convention:
        # Class 1: Target Drones (filename contains "target")
        # Class 0: Background / Birds (all other filenames)
        label = 1 if "target" in img_name.lower() else 0

        return img_tensor, label