import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from detectors.base_detector import ObjectDetector

class OwlViTDetector(ObjectDetector):
    """
    OWL-ViT object detection implementation.
    """

    def __init__(self, model_name="google/owlvit-base-patch32", device=None):
        """
        Initialize OWL-ViT detector.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)
        self.processor = OwlViTProcessor.from_pretrained(model_name)

    def detect(self, image, query_texts=None, confidence_threshold=0.03):
        """
        Detect objects in the given image using OWL-ViT.
        """
        # Load image
        image_data = self._load_image(image)
        if query_texts is None:
            query_texts = ["bottle", "cup", "book"]

        # Process inputs
        inputs = self.processor(text=[query_texts], images=image_data, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get bounding boxes and labels
        target_sizes = torch.tensor([(image_data.height, image_data.width)], dtype=torch.float32).to(self.device)
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=confidence_threshold)[0]

        # Format detections
        detections = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            if score >= confidence_threshold:
                detections.append({
                    'box': [round(i, 2) for i in box.tolist()],
                    'label': query_texts[label],
                    'score': round(score.item(), 3)
                })

        # Annotate image
        annotated_image = self._annotate_image(image_data, detections)
        return detections, annotated_image

    def _annotate_image(self, image, detections):
        """
        Draw bounding boxes and labels on the image and convert to NumPy array.
        """
        draw = ImageDraw.Draw(image)
        for obj in detections:
            x1, y1, x2, y2 = obj['box']
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 10), f"{obj['label']}: {obj['score']}", fill="red")

        # Convert PIL Image to NumPy array (RGB format)
        return np.array(image)


    def _load_image(self, image):
        """
        Load image from various sources.
        """
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, str):
            return Image.open(image).convert("RGB")
        raise ValueError("Unsupported image type")

    def get_model_info(self):
        """
        Retrieve model information.
        """
        return {
            'name': self.model_name,
            'type': 'OWL-ViT',
            'device': self.device,
            'open_vocabulary': True
        }
