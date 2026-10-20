import torch
import torch.nn as nn
import torch.nn.functional as F

class ThermalCNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) designed to classify
    thermal target images (e.g., Drone vs. Bird).
    """
    def __init__(self):
        super(ThermalCNN, self).__init__()

        # Convolutional Layer 1: Takes 1 input channel (grayscale), outputs 16 feature maps
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Max Pooling Layer: Shrinks image dimensions by half, keeping the strongest highlights (hot spots)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 2: Takes 16 channels, outputs 32 deeper feature maps
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Fully Connected (Linear) Layers for final classification
        # Note: Our 128x128 image shrinks to 32x32 after passing through two max pooling layers (128 -> 64 -> 32)
        self.fc1 = nn.Linear(32 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, 2) # Output: 2 classes (0 = Drone, 1 = Bird)

    def forward(self, x):
        # Pass through Conv1 -> ReLU Activation -> Max Pooling
        x = self.pool(F.relu(self.conv1(x)))

        # Pass through Conv2 -> ReLU Activation -> Max Pooling
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the 3D multi-channel image grid into a 1D line of numbers for the linear layers
        x = x.view(-1, 32 * 32 * 32)

        # Pass through the decision layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    # Test instantiating the model and passing a dummy tensor through it
    model = ThermalCNN()
    dummy_input = torch.randn(1, 1, 128, 128) # 1 image, 1 channel, 128x128 pixels
    output = model(dummy_input)
    print("Model Architecture Test Successful.")
    print(f"Output shape (Batch Size, Classes): {output.shape}")