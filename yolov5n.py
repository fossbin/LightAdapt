import json
from torchvision import transforms
import random
import os
import torch
import numpy as np
import cv2
from ultralytics import YOLO
import time

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
    return img, img_path


def generate_perlin_noise(shape):
    """Generate Perlin-like noise pattern"""
    noise = np.zeros(shape[:2])
    scale = 20.0
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0

    for i in range(octaves):
        freq = scale * (lacunarity ** i)
        amp = persistence ** i

        # Ensure minimum size of 2x2 for noise layer
        noise_size_h = max(2, int(shape[0]/freq))
        noise_size_w = max(2, int(shape[1]/freq))

        noise_layer = np.random.rand(noise_size_h, noise_size_w)

        # Resize using correct dimension order (width, height)
        noise_layer = cv2.resize(noise_layer, (shape[1], shape[0]))
        noise += noise_layer * amp

    # Normalize the noise to [0, 1] range
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return torch.from_numpy(noise).float().to(device)


def generate_organic_pattern(shape):
    """Generate organic-looking pattern"""
    pattern = np.zeros(shape[:2])
    num_circles = random.randint(3, 7)

    for _ in range(num_circles):
        center = (random.randint(0, shape[0]),
                  random.randint(0, shape[1]))
        radius = random.randint(5, shape[0]//3)
        color = random.random()

        y, x = np.ogrid[:shape[0], :shape[1]]
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        mask = dist <= radius
        pattern[mask] = color

    return torch.from_numpy(pattern).float().to(device)


patch_size = 100
# Initialize patch with darker base values
patch = torch.randn((3, patch_size, patch_size),
                    requires_grad=True, device=device) * 100  # Reduced from 255
patch = torch.nn.Parameter(patch)
optimizer = torch.optim.Adam([patch], lr=0.01)

# Generate noise patterns
patch_size = 100
noise_patterns = {
    'perlin': generate_perlin_noise((patch_size, patch_size)),
    'organic': generate_organic_pattern((patch_size, patch_size)),
    'gradient': torch.linspace(0, 1, patch_size).view(1, -1).repeat(patch_size, 1).to(device)
}


def apply_patch(img_tensor, patch, x, y):
    patched_img = img_tensor.clone()
    normalized_patch = torch.clamp(patch, 0, 255) / 255.0

    try:
        # Create alpha mask for smooth blending
        alpha = torch.ones((patch_size, patch_size), device=device) * 0.7
        alpha_np = alpha.cpu().numpy()
        alpha = torch.from_numpy(cv2.GaussianBlur(
            alpha_np, (5, 5), 2)).to(device)

        # Handle correct dimension ordering
        if len(patched_img.shape) == 4:  # batch, channel, height, width
            for c in range(3):  # For each color channel
                region = patched_img[0, c, y:y+patch_size, x:x+patch_size]
                region = region * (1 - alpha) + normalized_patch[c] * alpha
                patched_img[0, c, y:y+patch_size, x:x+patch_size] = region
        else:
            print("Unexpected input tensor shape:", patched_img.shape)
            return img_tensor

        return patched_img
    except Exception as e:
        print(f"Error applying patch at position ({x}, {y}): {str(e)}")
        return img_tensor


def safe_adversarial_loss(predictions, patch):
    """Enhanced loss function for dark environments"""
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

        # Enhanced loss calculation
        confidence_loss = torch.mean(conf_scores.float() ** 2)

        # Add smoothness regularization
        smoothness_loss = torch.mean(torch.abs(patch[:, 1:] - patch[:, :-1])) + \
            torch.mean(torch.abs(patch[:, :, 1:] - patch[:, :, :-1]))

        # Combine losses
        loss = confidence_loss + 0.1 * smoothness_loss

        if not loss.requires_grad:
            loss = loss + (patch.sum() * 0)

        return loss
    except Exception as e:
        print(f"Error in loss calculation: {e}")
        return (patch.sum() * 0) - 1


def adjust_patch_for_low_light(frame, patch_np):
    """Enhanced dark environment adaptation with proper type handling"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    local_std = np.std(gray)

    # Ensure patch_np is float32 for calculations
    patch_np = patch_np.astype(np.float32)

    # Adaptive adjustments based on scene properties
    if avg_brightness < 30:  # Very dark
        alpha = random.uniform(0.4, 0.6)
        beta = random.randint(-20, -10)
        noise_strength = random.uniform(5, 15)
    elif avg_brightness < 50:  # Moderately dark
        alpha = random.uniform(0.6, 0.8)
        beta = random.randint(-10, 0)
        noise_strength = random.uniform(3, 8)
    else:  # Brighter conditions
        alpha = random.uniform(0.8, 1.0)
        beta = random.randint(0, 10)
        noise_strength = random.uniform(2, 5)

    # Apply base adjustments
    adjusted = cv2.convertScaleAbs(patch_np, alpha=alpha, beta=beta)
    adjusted = adjusted.astype(np.float32)

    # Create noise pattern with proper type
    noise_pattern = np.zeros_like(adjusted, dtype=np.float32)

    # Convert noise patterns to float32
    perlin = noise_patterns['perlin'].cpu().numpy().astype(np.float32)
    organic = noise_patterns['organic'].cpu().numpy().astype(np.float32)
    gradient = noise_patterns['gradient'].cpu().numpy().astype(np.float32)

    # Ensure noise patterns are 2D
    if len(perlin.shape) == 1:
        perlin = np.tile(perlin, (patch_size, 1))
    if len(organic.shape) == 1:
        organic = np.tile(organic, (patch_size, 1))
    if len(gradient.shape) == 1:
        gradient = np.tile(gradient, (patch_size, 1))

    # Blend different noise patterns with proper broadcasting
    for c in range(3):
        noise_component = (
            perlin * random.uniform(0.3, 0.5) +
            organic * random.uniform(0.2, 0.4) +
            gradient * random.uniform(0.1, 0.3)
        )
        noise_pattern[:, :, c] = noise_component

    # Scale noise based on local scene properties
    noise_pattern = noise_pattern * (noise_strength * (local_std / 128.0))

    # Apply temporal variation
    temporal_factor = (np.sin(time.time() * 2) * 0.5 + 0.5).astype(np.float32)
    noise_pattern *= temporal_factor

    # Convert to uint8 for addWeighted
    adjusted_uint8 = np.clip(adjusted, 0, 255).astype(np.uint8)
    noise_pattern_uint8 = np.clip(noise_pattern, 0, 255).astype(np.uint8)

    # Apply noise
    adjusted = cv2.addWeighted(
        adjusted_uint8, 1.0, noise_pattern_uint8, 0.3, 0)

    # Add subtle texture
    texture = cv2.GaussianBlur(adjusted, (3, 3), 0.5)
    adjusted = cv2.addWeighted(adjusted, 0.7, texture, 0.3, 0)

    # Final clip to ensure valid range
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

    return adjusted
    
def get_targeted_patch_position(frame, model):
    """
    Get the position to place the patch based on the detected objects.
    """
    results = model(frame)
    if len(results[0].boxes) == 0:
        return random.randint(50, frame.shape[1]-patch_size), random.randint(50, frame.shape[0]-patch_size)

    # Get the box with the highest confidence
    max_conf_idx = results[0].boxes.conf.argmax()
    box = results[0].boxes.xyxy[max_conf_idx].cpu().numpy()

    # Place the patch in the center of the detected object
    x = int((box[0] + box[2]) // 2 - patch_size // 2)
    y = int((box[1] + box[3]) // 2 - patch_size // 2)

    # Ensure the patch is within the frame boundaries
    x = max(50, min(x, frame.shape[1] - patch_size - 50))
    y = max(50, min(y, frame.shape[0] - patch_size - 50))

    return x, y


cap = cv2.VideoCapture(0)  # Use 0 for default webcam

# Check if the video capture is successfully opened
if not cap.isOpened():
    print("Error: Could not open video capture device")
    exit()

# Set video properties (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# Main processing loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Get targeted position for the patch
        x, y = get_targeted_patch_position(frame, model)

        # Convert patch to numpy array if it's a tensor
        patch_np = patch.detach().cpu().numpy().transpose(1, 2, 0)

        # Adjust the patch for low-light conditions
        adjusted_patch = adjust_patch_for_low_light(frame, patch_np)

        # Apply the patch to the frame
        frame[y:y+patch_size, x:x+patch_size] = adjusted_patch

        # Get detections on the patched frame
        results = model(frame)
        annotated_frame = results[0].plot()

        cv2.imshow("Adversarial Attack", annotated_frame)

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    # Properly release resources
    cap.release()
    cv2.destroyAllWindows()
