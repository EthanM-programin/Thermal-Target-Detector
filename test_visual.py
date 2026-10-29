import os
import random
import cv2
import torch
from torchvision import transforms
from PIL import Image
from model import ThermalCNN
import config

def run_visual_test():
    print("[INFO] Initializing model for visual inference test...")
    device = torch.device("cpu")

    # 1. Load model architecture and trained weights
    model = ThermalCNN().to(device)
    weights_path = "thermal_cnn_weights.pth"
    if not os.path.exists(weights_path):
        print(f"[ERROR] Trained weights not found at {weights_path}. Please run train.py first.")
        return

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 2. Pick a random sample image from the local data directory.
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory not found at {data_dir}.")
        return

    images = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print("[ERROR] No images found in data directory.")
        return

    # 3. Loop through multiple test images in a single run (e.g., 5 samples)
    num_tests = 5
    print(f"[INFO] Starting evaluation batch of {num_tests} random images...")

    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    for i in range(num_tests):
        sample_img_name = random.choice(images)
        sample_img_path = os.path.join(data_dir, sample_img_name)

        # Apply grayscale + CLAHE simulation
        img_gray = cv2.imread(sample_img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_gray)

        # Tensor conversion
        img_pil = Image.fromarray(img_clahe)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            pred_class = predicted.item()
            conf_score = confidence.item() * 100

        # Display labels
        is_target = "target" in sample_img_name.lower()
        label_text = "TARGET DRONE" if pred_class == 1 else "BACKGROUND BIRD"
        status_color = (0, 0, 255) if pred_class == 1 else (255, 0, 0) # BGR

        print(f"\n[TEST {i+1}/{num_tests}] File: {sample_img_name}")
        print(f" -> Prediction: {label_text} | Confidence: {conf_score:.2f}%")
        print(f" -> Ground Truth Match -> Contains 'target': {is_target}")

        # OpenCV Render
        display_img = cv2.resize(img_clahe, (512, 512))
        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

        cv2.putText(display_img, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(display_img, f"Conf: {conf_score:.1f}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        file_display_str = sample_img_name if len(sample_img_name) < 35 else sample_img_name[:32] + "..."
        cv2.putText(display_img, f"File: {file_display_str}", (20, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow("Thermal Target Detector - Batch Test (Press any key for next)", display_img)
        # Wait for key press; pressing any key immediately loads the next test image
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("\n[SUCCESS] Completed batch evaluation session cleanly.")

if __name__ == "__main__":
    run_visual_test()