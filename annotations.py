from ultralytics import YOLO
import os


def generate_annotations(model_path, dataset_folder, output_folder, confidence_threshold=0.25):
    """
    Generate annotations using a pre-trained YOLOv8 model.
    Args:
        model_path (str): Path to the pre-trained YOLOv8 model (e.g., 'yolov8n.pt').
        dataset_folder (str): Path to the folder containing ExDark subfolders (e.g., 'Bicycle', 'Bottle').
        output_folder (str): Path to save the generated annotations.
        confidence_threshold (float): Confidence threshold for object detection (default: 0.25).
    """
    # Load the pre-trained YOLOv8 model
    model = YOLO(model_path)

    # Create output folder for annotations
    os.makedirs(output_folder, exist_ok=True)

    # ExDark class mapping (class names to IDs)
    exdark_class_mapping = {
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

    # COCO class names (as used by the pre-trained YOLOv8 model)
    coco_class_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]

    # Map COCO class names to ExDark class IDs
    coco_to_exdark_mapping = {
        'bicycle': exdark_class_mapping['Bicycle'],
        'boat': exdark_class_mapping['Boat'],
        'bottle': exdark_class_mapping['Bottle'],
        'bus': exdark_class_mapping['Bus'],
        'car': exdark_class_mapping['Car'],
        'cat': exdark_class_mapping['Cat'],
        'chair': exdark_class_mapping['Chair'],
        'cup': exdark_class_mapping['Cup'],
        'dog': exdark_class_mapping['Dog'],
        'motorcycle': exdark_class_mapping['Motorbike'],
        'person': exdark_class_mapping['People'],
        'dining table': exdark_class_mapping['Table']
    }

    # Traverse all subfolders in the dataset folder
    for category in os.listdir(dataset_folder):
        category_path = os.path.join(dataset_folder, category)
        if os.path.isdir(category_path):
            print(f"Processing category: {category}")

            # Create a subfolder in the output folder for this category
            output_category_folder = os.path.join(output_folder, category)
            os.makedirs(output_category_folder, exist_ok=True)

            # Process each image in the category folder
            for image_name in os.listdir(category_path):
                if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(category_path, image_name)
                    print(f"Processing image: {image_path}")

                    # Run inference on the image with a lower confidence threshold
                    results = model(image_path, conf=confidence_threshold)

                    # Save annotations in YOLO format
                    annotation_path = os.path.join(
                        output_category_folder, os.path.splitext(image_name)[0] + '.txt')
                    with open(annotation_path, 'w') as f:
                        for result in results:
                            for box in result.boxes:
                                # Get class ID (COCO class ID)
                                coco_class_id = int(box.cls)
                                coco_class_name = coco_class_names[coco_class_id]

                                # Check if the detected class is in the ExDark dataset
                                if coco_class_name in coco_to_exdark_mapping:
                                    # Map COCO class ID to ExDark class ID
                                    exdark_class_id = coco_to_exdark_mapping[coco_class_name]

                                    # Get bounding box coordinates (normalized)
                                    x_center, y_center, width, height = box.xywhn[0].tolist(
                                    )

                                    # Get confidence score
                                    confidence = float(box.conf)

                                    # Write to annotation file in YOLO format: class_id x_center y_center width height
                                    f.write(
                                        f"{exdark_class_id} {x_center} {y_center} {width} {height}\n")
                                    print(
                                        f"Detected: class={coco_class_name} (ExDark ID={exdark_class_id}), confidence={confidence:.2f}, bbox=({x_center}, {y_center}, {width}, {height})")

                    print(f"Generated annotations for: {image_name}")


# Call this function to generate annotations
generate_annotations(
    model_path="yolov8n.pt",  # Path to the pre-trained YOLOv8 model
    # Folder containing ExDark subfolders (e.g., 'Bicycle', 'Bottle')
    dataset_folder="ExDark",
    output_folder="annotations",  # Folder to save generated annotations
    confidence_threshold=0.25  # Lower confidence threshold for better detection
)
