import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data_loaders
from model import ThermalCNN

def train_model():
    print("Initializing Thermal CNN Training Pipeline...")

    # 1. Get our training and validation data loaders (Batch size 32, 1000 total samples)
    train_loader, val_loader = get_data_loaders(batch_size=32, num_samples=1000)

    # 2. Instantiate our model, loss function, and optimizer
    model = ThermalCNN()

    # CrossEntropyLoss is standard for multi-class/binary classification
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer updates the model's weights based on gradients
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    print(f"Starting training for {num_epochs} epochs...\n")

    for epoch in range(num_epochs):
        model.train() # Put model in training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Loop through batches of images and labels
        for images, labels in train_loader:
            # Zero out the gradient buffers from the previous step
            optimizer.zero_grad()

            # Forward pass: Compute predictions by passing images through the model
            outputs = model(images)

            # Calculate loss (how wrong the model's predictions were)
            loss = criterion(outputs, labels)

            # Backward pass: Compute gradients
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item()

            # Track training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct_train / total_train

        # Validation phase
        model.eval() # Put model in evaluation mode
        correct_val = 0
        total_val = 0

        with torch.no_grad(): # Turn off gradient tracking during validation to save memory
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = 100 * correct_val / total_val

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"| Loss: {epoch_loss:.4f} "
              f"| Train Accuracy: {epoch_acc:.2f}% "
              f"| Validation Accuracy: {val_acc:.2f}%")

    # 3. Save the trained model weights to disk
    torch.save(model.state_dict(), "thermal_cnn_weights.pth")
    print("\nTraining Complete. Model weights saved to 'thermal_cnn_weights.pth'.")

if __name__ == "__main__":
    train_model()