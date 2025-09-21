# Nano-object-detection-model
This project demonstrates real-time object detection using the Ultralytics YOLOv8  deep learning model. It takes an input image, detects multiple objects (people, vehicles, traffic lights, etc.), and saves an annotated output image with bounding boxes and confidence scores.
# Install ultralytics if not already done:
# pip install ultralytics

from ultralytics import YOLO
import os
execution_path = os.getcwd()
model = YOLO("yolov8n.pt") 
input_image_path = os.path.join(execution_path, "image.jpg")
output_image_path = os.path.join(execution_path, "Imagenew.jpg")
results = model(input_image_path)
results[0].save(save_dir=execution_path)
for result in results:
    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        probability = float(box.conf[0]) * 100
        print(class_name, " : ", round(probability, 2), "%")
