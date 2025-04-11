import torch
from ultralytics import YOLOWorld
import time
import cv2

image = cv2.imread("C:/Users/Lukas Hajduch/Downloads/BachelorThesis/data/images/IMG_20250316_210542_324.jpg")[..., ::-1]

for device in ["cpu", "cuda"]:
    print(f"\nTesting on {device.upper()}...")
    model = YOLOWorld("yolov8s-worldv2").to(device)
    model.eval()

    for _ in range(2):  # run twice to see warmup effect
        start = time.time()
        result = model.predict(image)[0]
        end = time.time()
        print(f"Time on {device.upper()}: {end - start:.2f} seconds")