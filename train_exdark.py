import yaml
from ultralytics import YOLO
import os
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import torch

# Add memory optimizations
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.8)  # Limit GPU memory usage
# Enable cuDNN benchmarking for faster training
torch.backends.cudnn.benchmark = True
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Debugging CUDA operations


def create_yolo_dataset(exdark_path, output_path):
    """
    Convert ExDark dataset to YOLO format from category subfolders.
    """
    # Create directory structure
    os.makedirs(f"{output_path}/images/train", exist_ok=True)
    os.makedirs(f"{output_path}/images/val", exist_ok=True)
    os.makedirs(f"{output_path}/labels/train", exist_ok=True)
    os.makedirs(f"{output_path}/labels/val", exist_ok=True)

    # Class mapping for YOLO format
    class_mapping = {
        'Bicycle': 0,
        'Boat': 1,
        'Bottle': 2,
        'Bus': 3,
        'Car': 4,
        'Cat': 5,
        'Chair': 6,
        'Cup': 7,
        'Dog': 8,
        'Motorbike': 9,
        'People': 10,
        'Table': 11
    }

    # Collect all images with their categories
    image_files = []
    for category in os.listdir(exdark_path):
        category_path = os.path.join(exdark_path, category)
        if os.path.isdir(category_path):
            class_id = class_mapping.get(category)
            if class_id is not None:  # Only process if category is in our mapping
                for img in os.listdir(category_path):
                    if img.lower().endswith(('.jpg', '.png', '.jpeg')):
                        image_files.append({
                            'path': os.path.join(category_path, img),
                            'category': category,
                            'class_id': class_id
                        })

    if not image_files:
        raise Exception("No images found in the dataset directory!")

    # Split into train/val
    train_files, val_files = train_test_split(
        image_files, test_size=0.2, random_state=42)

    def process_files(files, split):
        """Process and copy files to train/val splits."""
        for file_info in tqdm(files, desc=f"Processing {split} set"):
            img_path = file_info['path']
            class_id = file_info['class_id']

            # Generate unique filename to avoid conflicts
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]

            # Copy image
            dest_img_path = f"{output_path}/images/{split}/{img_name}"
            shutil.copy2(img_path, dest_img_path)

            # Create YOLO format label (dummy bounding box for demonstration)
            label_content = f"{class_id} 0.5 0.5 0.5 0.5\n"

            # Save label file
            with open(f"{output_path}/labels/{split}/{base_name}.txt", 'w') as f:
                f.write(label_content)

    process_files(train_files, "train")
    process_files(val_files, "val")

    return len(train_files), len(val_files)


def create_dataset_yaml(output_path):
    """
    Create YOLO dataset.yaml with ExDark classes.
    """
    yaml_content = {
        'path': os.path.abspath(output_path),  # Use absolute path
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'Bicycle',
            1: 'Boat',
            2: 'Bottle',
            3: 'Bus',
            4: 'Car',
            5: 'Cat',
            6: 'Chair',
            7: 'Cup',
            8: 'Dog',
            9: 'Motorbike',
            10: 'People',
            11: 'Table'
        }
    }

    yaml_path = f'{output_path}/dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    return yaml_path


def train_on_exdark(yaml_path):
    """
    Train YOLOv8 on ExDark dataset with optimized parameters for RTX 3050 6GB VRAM.
    """
    model_path = 'yolov8n.pt'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    # Load the YOLOv8n model
    model = YOLO(model_path)

    # Memory optimizations
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.95)  # Allow using more VRAM

    # Train with optimized parameters
    results = model.train(
        data=yaml_path,
        epochs=150,
        imgsz=384,
        batch=16,
        patience=50,
        device=0,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.01,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.1,
        nbs=64,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=45.0,
        translate=0.1,
        scale=0.5,
        shear=10.0,
        perspective=0.0001,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.5,
        copy_paste=0.3,
        auto_augment='randaugment',
        erasing=0.4,
        cache=True,
        save_period=5,
        workers=8,
        overlap_mask=True,
        max_det=300,
        amp=True,
        profile=True,
        # Multi-scale training
        multi_scale=True,
        scales=[0.8, 1.0, 1.2],
    )

    # Save the final model
    model.save('exdark_yolov8n_optimized.pt')
    return results


if __name__ == "__main__":
    # Set your paths
    EXDARK_PATH = "ExDark"  # Your ExDark folder containing category subfolders
    OUTPUT_PATH = "processed_exdark"

    try:
        # Process dataset
        print("Creating YOLO format dataset...")
        train_count, val_count = create_yolo_dataset(EXDARK_PATH, OUTPUT_PATH)
        print(
            f"Dataset split complete: {train_count} training images, {val_count} validation images")

        # Create yaml
        print("Creating dataset.yaml...")
        yaml_path = create_dataset_yaml(OUTPUT_PATH)
        print(f"Dataset configuration saved to {yaml_path}")

        # Train model
        print("Starting training...")
        results = train_on_exdark(yaml_path)
        print("Training complete!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
