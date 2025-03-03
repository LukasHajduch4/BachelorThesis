import sys
sys.path.append("C:/Users/Lukas Hajduch/OneDrive/Dokumenty/Bachelor thesis/BachelorThesis/Detic")
import torch
import numpy as np
import cv2
from PIL import Image
import requests
import os
from detectors.base_detector import ObjectDetector

class DeticDetector(ObjectDetector):
    """
    Implementation of object detector based on DETIC (CLIP with object detection capabilities).
    DETIC extends CLIP to enable open-vocabulary object detection.
    """
    
    def __init__(self, model_name="detic_centernet2_swin-b", device=None):
        """
        Initialize DETIC detector.
        
        Args:
            model_name: Model configuration to use
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        # Import the required libraries
        try:
            from detectron2.config import get_cfg
            from detectron2.projects.deeplab import add_deeplab_config
            from detic.config import add_detic_config
            from detic.modeling.text.text_encoder import build_text_encoder
            from detic.modeling.utils import reset_cls_test
        except ImportError:
            raise ImportError("Please install Detectron2 and DETIC: pip install 'git+https://github.com/facebookresearch/detectron2.git' and follow DETIC installation instructions")
            
        # Setting up DETIC configuration
        self.cfg = get_cfg()
        add_deeplab_config(self.cfg)
        add_detic_config(self.cfg)
        self.cfg.merge_from_file(f"configs/{model_name}.yaml")
        self.cfg.MODEL.WEIGHTS = f"models/{model_name}.pth"
        self.cfg.MODEL.DEVICE = self.device
        
        # Load CLIP text encoder for open vocabulary
        self.text_encoder = build_text_encoder(self.cfg)
        self.text_encoder.to(self.device)
        
        # Initialize predictor from Detectron2
        from detectron2.engine import DefaultPredictor
        self.model = DefaultPredictor(self.cfg)
        
        # ImageNet vocabulary by default
        self.vocabulary = "custom"
        
    def detect(self, image, query_texts=None, confidence_threshold=0.5):
        """
        Detect objects in the image.
        
        Args:
            image: Input image (numpy.ndarray or path to file)
            query_texts: Optional list of text descriptions for open-vocabulary detection
            confidence_threshold: Confidence threshold for filtering detections
            
        Returns:
            list: List of detected objects with keys:
                - box: Bounding box [x1, y1, x2, y2]
                - label: Class name
                - score: Confidence score
            numpy.ndarray: Annotated image with detections
        """
        # Load image
        image_data = self._load_image(image)
        
        # Process query_texts for open vocabulary
        if query_texts and len(query_texts) > 0:
            text_features = self.text_encoder(query_texts)
            custom_vocab = query_texts
            reset_cls_test(self.model.model, text_features, custom_vocab)
        
        # Run detection
        outputs = self.model(image_data)
        
        # Extract results
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
        scores = instances.scores.numpy() if instances.has("scores") else []
        labels = instances.pred_classes.numpy() if instances.has("pred_classes") else []
        
        # Create detections list
        detections = []
        for box, score, label_idx in zip(boxes, scores, labels):
            if score >= confidence_threshold:
                # Get class name from either custom vocabulary or model's vocabulary
                if query_texts and len(query_texts) > label_idx:
                    label = query_texts[label_idx]
                else:
                    label = f"class_{label_idx}"
                
                detections.append({
                    'box': box.tolist(),
                    'label': label,
                    'score': float(score)
                })
        
        # Draw annotations
        annotated_image = image_data.copy()
        for obj in detections:
            x1, y1, x2, y2 = map(int, obj['box'])
            label_text = f"{obj['label']}: {obj['score']:.2f}"
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return detections, annotated_image
    
    def _load_image(self, image):
        """
        Load image from various sources.
        """
        if isinstance(image, np.ndarray):
            return image
        if isinstance(image, str):
            if image.startswith('http'):
                response = requests.get(image, stream=True)
                return np.array(Image.open(response.raw).convert("RGB"))
            else:
                return cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        raise ValueError("Unsupported image type")
    
    def get_model_info(self):
        """
        Get information about the model.
        """
        return {
            'name': self.model_name,
            'type': 'DETIC',
            'device': self.device,
            'open_vocabulary': True,
            'description': 'DETIC extends CLIP for open-vocabulary object detection'
        }