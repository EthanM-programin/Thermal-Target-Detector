import cv2
import torch
import numpy as np
import config

def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """Applies real-time CLAHE enhancement and formats frame for PyTorch model inference."""
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT, 
        tileGridSize=config.CLAHE_GRID_SIZE
    )
    enhanced = clahe.apply(gray)
    
    resized = cv2.resize(enhanced, config.IMAGE_SIZE)
    
    tensor = torch.from_numpy(resized).float() / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    return tensor

def run_inference(model: torch.nn.Module, frame: np.ndarray, device: torch.device):
    """Executes target identification on video frame and draws real-time detection overlays."""
    input_tensor = preprocess_frame(frame).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    conf_score = confidence.item()
    class_id = predicted_class.item()

    # Draw bounding overlay if target confidence exceeds configured threshold
    if class_id == 1 and conf_score >= config.CONFIDENCE_THRESHOLD:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (10, 10), (w - 10, h - 10), (0, 255, 0), 2)
        label = f"Target Detected: {conf_score * 100:.2f}%"
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame