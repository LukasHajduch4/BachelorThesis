import torch
import numpy as np
import cv2
from PIL import Image
import requests
import os
from detectors.base_detector import ObjectDetector
import clip
from segment_anything import sam_model_registry, SamPredictor

class SamClipDetector(ObjectDetector):
    """
    Implementation of object detector combining Segment Anything Model (SAM) with CLIP.
    SAM is used for segmentation while CLIP enables semantic understanding and classification.
    """
    
    def __init__(self, 
                 sam_checkpoint="sam_vit_h_4b8939.pth",
                 sam_model_type="vit_h",
                 clip_model_name="ViT-B/32",
                 device=None):
        """
        Initialize SAM+CLIP detector.
        
        Args:
            sam_checkpoint: Path to SAM model checkpoint
            sam_model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            clip_model_name: CLIP model name
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sam_checkpoint = sam_checkpoint
        self.sam_model_type = sam_model_type
        self.clip_model_name = clip_model_name
        
        # Load SAM model
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to(self.device)
        self.sam_predictor = SamPredictor(sam)
        
        # Load CLIP model
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
    
    def detect(self, image, query_texts=None, confidence_threshold=0.5):
        """
        Detect objects in the image.
        
        Args:
            image: Input image (numpy.ndarray or path to file)
            query_texts: List of text descriptions for classification
            confidence_threshold: Confidence threshold for filtering detections
            
        Returns:
            list: List of detected objects with keys:
                - box: Bounding box [x1, y1, x2, y2]
                - label: Class name
                - score: Confidence score
                - mask: Segmentation mask
            numpy.ndarray: Annotated image with detections
        """
        # Load image
        image_data = self._load_image(image)
        
        # Ensure we have query texts
        if not query_texts or len(query_texts) == 0:
            query_texts = ["book", "cup", "bottle", "pen", "chair"]  # Default queries if none provided
        
        # Step 1: Generate automatic mask proposals with SAM
        self.sam_predictor.set_image(image_data)
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            multimask_output=True,
            box=None
        )
        
        # Step 2: Extract image regions based on masks
        detections = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            # Skip low confidence masks from SAM
            if score < 0.7:  # SAM confidence threshold
                continue
                
            # Extract bounding box from mask
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
                
            x1, x2 = np.min(x_indices), np.max(x_indices)
            y1, y2 = np.min(y_indices), np.max(y_indices)
            
            # Expand box slightly
            h, w = image_data.shape[:2]
            x1 = max(0, x1 - 5)
            y1 = max(0, y1 - 5)
            x2 = min(w - 1, x2 + 5)
            y2 = min(h - 1, y2 + 5)
            
            # Skip tiny regions
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue
                
            # Extract region of interest
            roi = image_data.copy()
            roi[~mask] = 0  # Zero out pixels outside the mask
            roi = roi[y1:y2, x1:x2]
            
            # Step 3: Classify with CLIP
            clip_image = self.clip_preprocess(Image.fromarray(roi)).unsqueeze(0).to(self.device)
            text_inputs = clip.tokenize(query_texts).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(clip_image)
                text_features = self.clip_model.encode_text(text_inputs)
                
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity scores
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
            # Get the highest scoring class
            values, indices = similarity[0].topk(1)
            score = values[0].item()
            label_idx = indices[0].item()
            label = query_texts[label_idx]
            
            # Only add detection if confidence is high enough
            if score >= confidence_threshold:
                detections.append({
                    'box': [float(x1), float(y1), float(x2), float(y2)],
                    'label': label,
                    'score': score,
                    'mask': mask.copy()  # Include the segmentation mask
                })
        
        # Sort detections by confidence score (highest first)
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        # Draw annotations
        annotated_image = image_data.copy()
        
        # Create colored masks for visualization
        mask_image = np.zeros_like(annotated_image)
        
        for i, obj in enumerate(detections):
            # Generate a random color for each mask
            color = np.random.randint(0, 255, size=3, dtype=np.uint8)
            x1, y1, x2, y2 = map(int, obj['box'])
            
            # Draw the mask
            mask = obj['mask']
            colored_mask = np.zeros_like(annotated_image)
            colored_mask[mask] = color
            mask_image = cv2.addWeighted(mask_image, 1, colored_mask, 0.5, 0)
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color.tolist(), 2)
            
            # Draw label
            label_text = f"{obj['label']}: {obj['score']:.2f}"
            cv2.putText(annotated_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color.tolist(), 2)
        
        # Combine original image with mask
        final_image = cv2.addWeighted(annotated_image, 0.7, mask_image, 0.3, 0)
        
        return detections, final_image
    
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
            'name': f"SAM-{self.sam_model_type}_CLIP-{self.clip_model_name}",
            'type': 'SAM+CLIP',
            'device': self.device,
            'open_vocabulary': True,
            'description': 'Combination of Segment Anything Model (SAM) for segmentation and CLIP for classification'
        }