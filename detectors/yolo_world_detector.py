import torch
import cv2
import numpy as np
from PIL import Image
import requests
import os
from ultralytics import YOLOWorld
from detectors.base_detector import ObjectDetector

class YOLOWorldDetector(ObjectDetector):
    """
    Implementácia detektora objektov založeného na YOLO World.
    """
    
    def __init__(self, model_name="yolov8x-worldv2", device=None):
        """
        Inicializácia YOLO World detektora.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = YOLOWorld(model_name).to(self.device)
        
    def detect(self, image, query_texts=None, confidence_threshold=0.2, iou_threshold=0.5):
        """
        Detekcia objektov pomocou YOLO World.
        """
        image_data = self._load_image(image)
        query_texts = ["book", "cup", "bottle"]

        results = self.model.predict(image_data)[0]

        detections = []
        filtered_boxes = []
        
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            if conf >= confidence_threshold:
                class_id = int(cls)
                class_name = self.model.names[class_id]
                if class_name in query_texts:
                    # Check for overlapping boxes
                    keep = True
                    for existing_box in filtered_boxes:
                        iou = self._calculate_iou(box.tolist(), existing_box['box'])
                        if iou > iou_threshold and conf < existing_box['score']:
                            keep = False
                            break
                    if keep:
                        filtered_boxes.append({
                            'box': box.tolist(),
                            'label': class_name,
                            'score': float(conf)
                        })
        
        detections = filtered_boxes
        
        # Draw annotations
        annotated_image = image_data.copy()
        for obj in detections:
            x1, y1, x2, y2 = map(int, obj['box'])
            label_text = f"{obj['label']}: {obj['score']:.2f}"
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return detections, annotated_image
    
    def _calculate_iou(self, box1, box2):
        """
        Výpočet Intersection over Union (IoU) na odstránenie duplikátov.
        """
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2
        
        xi1 = max(x1, x1g)
        yi1 = max(y1, y1g)
        xi2 = min(x2, x2g)
        yi2 = min(y2, y2g)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2g - x1g) * (y2g - y1g)
        
        iou = inter_area / (box1_area + box2_area - inter_area)
        return iou
    
    def _load_image(self, image):
        """
        Načítanie obrazu z rôznych zdrojov.
        """
        if isinstance(image, np.ndarray):
            return image
        if isinstance(image, str):
            if image.startswith('http'):
                response = requests.get(image, stream=True)
                return np.array(Image.open(response.raw).convert("RGB"))
            else:
                return cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        raise ValueError("Nepodporovaný typ obrazu")
    
    def get_model_info(self):
        """
        Získanie informácií o modeli.
        """
        return {
            'name': self.model_name,
            'type': 'YOLO World',
            'device': self.device,
            'open_vocabulary': True
        }
