import torch
import cv2
from ultralytics import YOLO

# Check for CUDA availability
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    print("CUDA is not available. Using CPU instead.")

# Initialize model
model = YOLO("yolov8n.pt")
model.to(device)


def print_detection_info(results):
    """Print detection information"""
    boxes = results[0].boxes
    if len(boxes) > 0:
        print("\nDetections:")
        print(f"Number of detections: {len(boxes)}")
        for i, (cls, conf) in enumerate(zip(boxes.cls, boxes.conf)):
            class_name = model.names[int(cls)]
            print(f"  {i+1}. {class_name}: {conf:.2f}")
    else:
        print("\nNo detections")


# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture device")
    exit()

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

try:
    while True:
        # Process camera frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab camera frame")
            break

        # Get detections
        results = model(frame)
        annotated_frame = results[0].plot()

        # Print detection information
        print_detection_info(results)

        # Display the frame (using direct display like the reference code)
        cv2.imshow("Object Detection", annotated_frame)

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    cap.release()
    cv2.destroyAllWindows()
