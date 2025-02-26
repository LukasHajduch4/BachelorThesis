import cv2
import numpy as np
import os
import time

from detectors.yolo_world_detector import YOLOWorldDetector
from pointing.hand_detector import HandPointingDetector
from pointing.pointing_analyzer import PointingAnalyzer
from stereo.stereo_processor import StereoProcessor
from utils.visualization import Visualizer

class ObjectPointingRecognitionSystem:
    """
    Hlavná systémová trieda integrujúca všetky komponenty.
    """
    
    def __init__(self, detector_type='yolo_world', calibration_file='config/camera_params.json'):
        """
        Inicializácia systému.
        """
        if detector_type == 'yolo_world':
            self.object_detector = YOLOWorldDetector()
        else:
            raise ValueError(f"Nepodporovaný typ detektora: {detector_type}")
        
        self.hand_detector = HandPointingDetector()
        self.pointing_analyzer = PointingAnalyzer()
        self.stereo_processor = StereoProcessor(calibration_file)
        self.visualizer = Visualizer()
    
    def process_stereo_images(self, left_image, right_image, query_texts=None):
        """
        Spracovanie stereo obrazov.
        """
        results = {}
        visualizations = {}
        
        left_rectified, right_rectified = self.stereo_processor.rectify_images(left_image, right_image)
        visualizations['rectified'] = np.hstack((left_rectified, right_rectified))
        
        objects, _ = self.object_detector.detect(left_rectified, query_texts)
        results['objects'] = objects
        
        hand_landmarks, annotated_hand = self.hand_detector.detect_hands(left_rectified)
        visualizations['hand_detection'] = annotated_hand
        
        pointing_direction = self.pointing_analyzer.detect_pointing_direction(hand_landmarks) if hand_landmarks else None
        results['pointing_direction'] = pointing_direction
        
        disparity, disparity_visual = self.stereo_processor.compute_disparity(left_rectified, right_rectified)
        visualizations['disparity'] = disparity_visual
        
        pointed_object_idx = self.pointing_analyzer.find_pointed_object(pointing_direction, objects, left_rectified.shape) if pointing_direction else None
        results['pointed_object'] = objects[pointed_object_idx] if pointed_object_idx is not None else None
        
        visualizations['final'] = self.visualizer.visualize_pointing_analysis(left_rectified, pointing_direction, objects, pointed_object_idx)
        
        return results, visualizations