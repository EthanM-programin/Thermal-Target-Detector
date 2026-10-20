import cv2
import torch
import numpy as np
from model import ThermalCNN
from data_loader import ThermalDataset

def run_detection_pipeline():
    print("Initializing Thermal Detection & Surveillance Stream...")

    # 1. Load the trained model architecture and our saved weights
    model = ThermalCNN()
    model.load_state_dict(torch.load("thermal_cnn_weights.pth"))
    model.eval() # Set model to evaluation mode (turns off training adjustments)

    # Initialize our dataset helper to draw sythetic frames from
    dataset = ThermalDataset(num_samples=10, img_size=(128,128))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    print("\nStarting live inference window. Press 'q' in the video window to exit.\n")

    # Simulate a live sensor stream loop
    for i in range(50):
        # Grab a random raw thermal frame and its true label from our dataset logic
        true_label = np.random.randint(0, 2)
        raw_frame = dataset._generate_synthetic_thermal_frame(true_label)

        # Apply CLAHE enhancement
        enhanced_frame = clahe.apply(raw_frame)

        # Prepare the frame into a PyTorch Tensor: shape (1, 1, 128, 128)
        tensor_frame = torch.tensor(enhanced_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

        # Run Inference: Pass tensor through our trained CNN
        with torch.no_grad():
            outputs = model(tensor_frame)
            probabilites = torch.softmax(outputs, dim=1) # Convert raw scores to percentages
            confidence, predicted_idx = torch.max(probabilites, 1)

        pred_class = predicted_idx.item()
        conf_score = confidence.item() * 100

        # Format display text
        class_name = "DRONE" if pred_class == 0 else "BIRD"
        color = (0, 0, 255) if pred_class == 0 else (0, 255, 0) # Red for Drone threat, Green for Bird

        # Upscale the small 128x128 image so it's easy to see on screen using OpenCV
        display_frame = cv2.resize(enhanced_frame, (400, 400), interpolation=cv2.INTER_NEAREST)
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR) # Convert to color se we can draw colored boxes

        # Draw a target bounding box and text overlay using OpenCV
        cv2.rectangle(display_frame, (100, 100), (300, 300), color, 2)
        cv2.putText(display_frame, f"Target: {class_name}", (110, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(display_frame, f"Confidence: {conf_score:.1f}%", (110, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Show the live stream window
        cv2.imshow("NADP Thermal Target Survewillance", display_frame)

        # Pause for 500 milliseconds per frame (press 'q' to break early)
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Surveillance stream closed.")

if __name__ == "__main__":
    run_detection_pipeline()