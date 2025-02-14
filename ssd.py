import numpy as np
import cv2
import torch
import torchvision
import random
import os
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for inference.")

# Load SSD model
model = torchvision.models.detection.ssd300_vgg16(
    weights=torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT)

# Get in_channels (as corrected before)
in_channels_list = [model.head.classification_head.module_list[i].in_channels for i in range(
    len(model.head.classification_head.module_list))]

# Calculate num_anchors (this is still correct)
num_classes = 13  # Your number of ExDark classes
num_predictions_per_location = (
    model.head.classification_head.module_list[-1].out_channels) // num_classes
num_anchors = num_predictions_per_location

num_anchors_list = [num_anchors] * len(in_channels_list)


# Create new classification head (finally correct!)
model.head.classification_head = SSDClassificationHead(
    in_channels=in_channels_list,  # Use the list here
    num_anchors=num_anchors_list,    # Use the list here
    num_classes=num_classes
)

model.to(device)

# Custom Dataset for ExDark


class ExDarkDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_files = []

        for category_folder in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category_folder)
            if os.path.isdir(category_path):
                for file in os.listdir(category_path):
                    if file.endswith((".jpg", ".png")):
                        self.image_files.append(
                            os.path.join(category_path, file))

        if not self.image_files:
            raise FileNotFoundError(
                "No images found in the dataset directory.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Dummy target for training
        target = {
            'boxes': torch.zeros((0, 4), device=device),
            'labels': torch.zeros(0, dtype=torch.int64, device=device)
        }

        return image, target


# Initialize dataset and transforms
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])

# Path to the ExDark dataset
dataset_path = "D:/steve/LightAdapt/ExDark"  # Update this to your actual path
dataset = ExDarkDataset(dataset_path=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# Initialize adversarial patch
patch_size = 100
patch = torch.randn((3, patch_size, patch_size),
                    requires_grad=True, device=device) * 255
patch = torch.nn.Parameter(patch)
optimizer = torch.optim.Adam([patch], lr=0.01)


def apply_patch(img_tensor, patch, x, y):
    patched_img = img_tensor.clone()
    normalized_patch = torch.clamp(patch, 0, 255) / 255.0
    try:
        patched_img[:, :, y:y+patch.shape[1],
                    x:x+patch.shape[2]] = normalized_patch
        return patched_img
    except:
        print(f"Error applying patch at position ({x}, {y})")
        return img_tensor


def safe_adversarial_loss(predictions):
    try:
        if not predictions:
            return torch.tensor(0.0, device=device, requires_grad=True)

        scores = predictions[0]['scores']
        if len(scores) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -torch.mean(scores + 1e-7)

        if not loss.requires_grad:
            loss = loss + (patch.sum() * 0)

        return loss
    except Exception as e:
        print(f"Error in loss calculation: {e}")
        return (patch.sum() * 0) - 1


# Training loop with results tracking
results = {"clean": [], "adversarial": []}

print("Starting training loop...")
for epoch in range(100):
    try:
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)

            # Clean predictions
            with torch.no_grad():
                clean_output = model(images)
                clean_detections = [len(output['boxes'])
                                    for output in clean_output]

            for i, num_detected in enumerate(clean_detections):
                results["clean"].append({
                    "detected": num_detected,
                    "confidence": clean_output[i]['scores'].cpu().tolist(),
                    "image": dataset.image_files[batch_idx * dataloader.batch_size + i]
                })

            # Apply adversarial patch
            x, y = random.randint(0, 200), random.randint(0, 200)
            patched_images = apply_patch(images, patch, x, y)

            # Forward pass with patched images
            adv_output = model(patched_images)
            loss = safe_adversarial_loss(adv_output)

            # Store adversarial results
            adv_detections = [len(output['boxes']) for output in adv_output]
            for i, num_detected in enumerate(adv_detections):
                results["adversarial"].append({
                    "detected": num_detected,
                    "confidence": adv_output[i]['scores'].cpu().tolist(),
                    "image": dataset.image_files[batch_idx * dataloader.batch_size + i]
                })

            # Optimize
            optimizer.zero_grad()
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    patch.data.clamp_(0, 255)

                print(
                    f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}, Detections: {sum(adv_detections)}")
            else:
                print(
                    f"Skipping optimization for batch {batch_idx} - no gradients")

    except Exception as e:
        print(f"Error in epoch {epoch}: {e}")
        continue

# Save results
print("Saving results...")
with open("ssd_detection_results.json", "w") as f:
    json.dump(results, f, indent=4)

# Convert patch for visualization
patch_np = (patch.detach().cpu().numpy() * 255).astype(np.uint8)
patch_np = np.transpose(patch_np, (1, 2, 0))

# Real-time webcam testing


def adjust_patch_based_on_lighting(frame, patch_np):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)

    if avg_brightness < 50:
        return cv2.addWeighted(patch_np, 0.7, np.zeros_like(patch_np), 0.3, 0)
    elif avg_brightness > 150:
        return cv2.addWeighted(patch_np, 1.3, np.zeros_like(patch_np), -0.3, 0)
    return patch_np.copy()


print("Starting webcam testing...")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to SSD input size
    frame = cv2.resize(frame, (300, 300))

    # Apply adjusted patch
    adjusted_patch = adjust_patch_based_on_lighting(frame, patch_np)
    x, y = random.randint(
        0, frame.shape[1]-patch_size), random.randint(0, frame.shape[0]-patch_size)
    frame[y:y+patch_size, x:x+patch_size] = adjusted_patch

    # Convert to tensor and get predictions
    frame_tensor = transform(Image.fromarray(cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(frame_tensor)[0]

    # Draw predictions
    for i, box in enumerate(predictions['boxes']):
        if predictions['scores'][i] > 0.5:
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("SSD Adversarial Attack", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
