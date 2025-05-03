import cv2
from ultralytics import YOLO
import os

model = YOLO("best.pt")
vehicle_classes = {2: "car", 4: "motorbike", 1: "bus", 6: "truck"}
pic_path = 'datasets/test/images'

output_folder = 'marked_images'
os.makedirs(output_folder, exist_ok=True)

""" This code goes through the provided test images, counts and marks the cars, bikes, buses and trucks.
    Processed images are outputed to the marked_images folder 
    YOLO was trained from the v8nano and fine-tuned using the dataset:
    https://universe.roboflow.com/redlightrunningdection/traffic-detection-sutq6 """

total_counts = {
    'car': 0,
    'motorbike': 0,
    'bus': 0,
    'truck': 0
    }

for filename in os.listdir(pic_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):

        image_path = os.path.join(pic_path, filename)
        img = cv2.imread(image_path)

        results = model.predict(source=img, conf=0.3)[0]

        counts = {
                    'car': 0,
                    'motorbike': 0,
                    'bus': 0,
                    'truck': 0
                }

        # Go through every box
        for box in results.boxes:
            cls_id = int(box.cls)
            if cls_id in vehicle_classes:
                # Update both counters for vehicle
                vehicle_type = vehicle_classes[cls_id]
                counts[vehicle_type] += 1
                total_counts[vehicle_type] += 1

                # Draws bounding boxes
                coords = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
                cv2.putText(img, vehicle_type, (coords[0], coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        # Save the images with bounding boxes
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, img)

        print(f"{save_path} counts: {counts}")

print(f"Total vehicles counted: {total_counts}")