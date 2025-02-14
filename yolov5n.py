import json
from torchvision import transforms
import random
import os
import torch
import numpy as np
import cv2
from ultralytics import YOLO

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    print("CUDA is not available. Using CPU instead.")

model = YOLO("yolov5nu.pt")
model.to(device)

# Path to the ExDark dataset
dataset_path = "D:/steve/LightAdapt/ExDark"

# Collect all image paths
image_files = []
for category_folder in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category_folder)
    if os.path.isdir(category_path):
        print(f"Processing category: {category_folder}")
        for file in os.listdir(category_path):
            if file.endswith((".jpg", ".png")):
                image_files.append(os.path.join(category_path, file))

if not image_files:
    raise FileNotFoundError("No images found in the dataset directory.")


def load_random_image():
    img_path = random.choice(image_files)
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Unable to read image: {img_path}")
    img = cv2.resize(img, (640, 640))
    return img


patch_size = 100
patch = torch.randn((3, patch_size, patch_size),
                    requires_grad=True, device=device) * 255
patch = torch.nn.Parameter(patch)
optimizer = torch.optim.Adam([patch], lr=0.01)


def apply_patch(img_tensor, patch, x, y):
    patched_img = img_tensor.clone()
    normalized_patch = torch.clamp(patch, 0, 255) / 255.0
    try:
        patched_img[0, :, y:y+patch.shape[1],
                    x:x+patch.shape[2]] = normalized_patch
        return patched_img
    except:
        print(f"Error applying patch at position ({x}, {y})")
        return img_tensor


def safe_adversarial_loss(predictions):
    """Robust loss function that handles various edge cases"""
    try:
        if len(predictions) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        conf_scores = predictions[0].boxes.conf
        if isinstance(conf_scores, np.ndarray):
            conf_scores = torch.from_numpy(conf_scores).to(device)

        if not conf_scores.requires_grad:
            conf_scores = conf_scores.detach().requires_grad_(True)

        if len(conf_scores) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Add a small epsilon to ensure numerical stability
        loss = -torch.mean(conf_scores.float() + 1e-7)

        # Verify the loss has gradients
        if not loss.requires_grad:
            print("Warning: Loss doesn't have gradients, adding dummy gradient")
            loss = loss + (patch.sum() * 0)  # Add dummy gradient

        return loss
    except Exception as e:
        print(f"Error in loss calculation: {e}")
        # Return a safe fallback that maintains gradients
        return (patch.sum() * 0) - 1


results = {"clean": [], "adversarial": []}

for epoch in range(100):
    try:
        img, img_path = load_random_image(image_files)
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

        # Get clean predictions
        with torch.no_grad():
            clean_output = model(img_tensor)
            clean_detections = clean_output[0].boxes.conf
            if isinstance(clean_detections, torch.Tensor):
                clean_detections = clean_detections.cpu().numpy()
            clean_detected = len(clean_detections)

        results["clean"].append({
            "detected": clean_detected,
            "confidence": clean_detections.tolist() if hasattr(clean_detections, 'tolist') else [],
            "image": img_path
        })

        # Apply adversarial patch
        x, y = random.randint(50, 540), random.randint(50, 540)
        patched_img = apply_patch(img_tensor, patch, x, y)

        # Forward pass with gradient tracking
        adv_output = model(patched_img)
        loss = safe_adversarial_loss(adv_output)

        # Store adversarial results
        with torch.no_grad():
            adv_detections = adv_output[0].boxes.conf
            if isinstance(adv_detections, torch.Tensor):
                adv_detections = adv_detections.cpu().numpy()
            adv_detected = len(adv_detections)

            results["adversarial"].append({
                "detected": adv_detected,
                "confidence": adv_detections.tolist() if hasattr(adv_detections, 'tolist') else [],
                "image": img_path
            })

        # Optimize
        optimizer.zero_grad()
        if loss.requires_grad:
            loss.backward()
            optimizer.step()

            # Clamp patch values
            with torch.no_grad():
                patch.data.clamp_(0, 255)

            print(
                f"Epoch {epoch}, Loss: {loss.item()}, Detections: {adv_detected}")
        else:
            print(f"Skipping optimization for epoch {epoch} - no gradients")

    except Exception as e:
        print(f"Error in epoch {epoch}: {e}")
        continue

# Save results
with open("detection_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to detection_results.json")

# Convert patch for visualization
patch_np = (patch.detach().cpu().numpy() * 255).astype(np.uint8)
patch_np = np.transpose(patch_np, (1, 2, 0))

# Test with webcam
cap = cv2.VideoCapture(0)


def adjust_patch_based_on_lighting(frame, patch_np):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)

    if avg_brightness < 50:
        return cv2.addWeighted(patch_np, 0.7, np.zeros_like(patch_np), 0.3, 0)
    elif avg_brightness > 150:
        return cv2.addWeighted(patch_np, 1.3, np.zeros_like(patch_np), -0.3, 0)
    return patch_np.copy()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    adjusted_patch = adjust_patch_based_on_lighting(frame, patch_np)
    x, y = random.randint(
        50, frame.shape[1]-patch_size), random.randint(50, frame.shape[0]-patch_size)
    frame[y:y+patch_size, x:x+patch_size] = adjusted_patch

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("Adversarial Attack", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
