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


def create_yolo_dataset(exdark_path, annotations_path, output_path):
    """
    Convert ExDark dataset to YOLO format from category subfolders.
    Args:
        exdark_path (str): Path to the ExDark dataset folder (with subfolders like Bicycle, Bottle, etc.).
        annotations_path (str): Path to the annotations folder (with subfolders like Bicycle, Bottle, etc.).
        output_path (str): Path to save the processed dataset in YOLO format.
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
            category = file_info['category']

            # Generate unique filename to avoid conflicts
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]

            # Copy image
            dest_img_path = f"{output_path}/images/{split}/{img_name}"
            shutil.copy2(img_path, dest_img_path)

            # Copy corresponding annotation file
            annotation_path = os.path.join(
                annotations_path, category, base_name + '.txt')
            if os.path.exists(annotation_path):
                dest_annotation_path = f"{output_path}/labels/{split}/{base_name}.txt"
                shutil.copy2(annotation_path, dest_annotation_path)
            else:
                print(f"Warning: No annotation file found for {img_name}")

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
    Train YOLOv8 on ExDark dataset with optimized parameters.
    """
    # Load pretrained YOLOv8 model
    model = YOLO('yolov8n.pt')  # You can switch to yolov8s.pt or yolov8m.pt

    # Train the model with improved hyperparameters
    results = model.train(
        data=yaml_path,
        epochs=50,  # Reduced epochs
        imgsz=640,  # Larger image size
        batch=16,  # Increased batch size
        patience=10,  # Early stopping
        device=0 if torch.cuda.is_available() else 'cpu',
        optimizer='SGD',  # Use SGD optimizer
        lr0=0.001,  # Lower learning rate
        lrf=0.1,  # Learning rate scheduler
        mosaic=1.0,  # Enable mosaic augmentation
        mixup=0.1,  # Enable mixup augmentation
        cache=False,  # Disable caching to save memory
        amp=True,  # Mixed precision training
        workers=4,  # Increased workers for faster data loading
        overlap_mask=True,  # Enable overlapping masks
        max_det=300,  # Maximum number of detections per image
        profile=True,  # Profile CUDA memory usage
    )

    # Save the final model
    model.save('exdark_yolov8n_improved.pt')
    return results


if __name__ == "__main__":
    # Set your paths
    EXDARK_PATH = "d:/steve/LightAdapt/ExDark"  # Your ExDark folder containing category subfolders
    # Folder containing annotations (with subfolders like Bicycle, Bottle, etc.)
    ANNOTATIONS_PATH = "d:/steve/LightAdapt/annotations"
    OUTPUT_PATH = "processed_exdark"

    try:
        # Process dataset
        print("Creating YOLO format dataset...")
        train_count, val_count = create_yolo_dataset(
            EXDARK_PATH, ANNOTATIONS_PATH, OUTPUT_PATH)
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
