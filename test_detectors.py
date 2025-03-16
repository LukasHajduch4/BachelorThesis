import cv2
import argparse
import os
import time
import numpy as np
from detectors.yolo_world_detector import YOLOWorldDetector
from detectors.detic_detector import DeticDetector
from detectors.owlvit_detector import OwlViTDetector
from detectors.sam_clip_detector import SamClipDetector

def test_detector(detector, image_path, query_texts):
    """
    Test a specific detector with given image and queries.
    
    Args:
        detector: Initialized detector object
        image_path: Path to the image for testing
        query_texts: List of query texts for open-vocabulary detection
    """
    print(f"\nTesting {detector.get_model_info()['type']} detector...")
    
    # Start timing
    start_time = time.time()
    
    # Run detection
    detections, annotated_image = detector.detect(image_path, query_texts)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"Detection completed in {elapsed_time:.2f} seconds")
    print(f"Found {len(detections)} objects:")
    
    for i, obj in enumerate(detections):
        print(f"  {i+1}. {obj['label']} - Confidence: {obj['score']:.2f}")
    
    # Create output directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Save annotated image
    detector_type = detector.get_model_info()['type'].lower().replace('+', '_')
    output_path = f"outputs/{detector_type}_result.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    print(f"Saved annotated image to {output_path}")
    
    # Display image
    cv2.imshow(f"{detector.get_model_info()['type']} Detection Results", 
               cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

def main():
    """
    Main function to test all detectors.
    """
    parser = argparse.ArgumentParser(description="Test different object detectors")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--queries", type=str, nargs='+', default=["book", "cup", "bottle", "chair", "person"], 
                        help="Query texts for detection")
    parser.add_argument("--detectors", type=str, nargs='+', default=["yolo", "detic", "owlvit", "sam_clip"],
                        help="Detectors to test (yolo, detic, owlvit, sam_clip)")
    args = parser.parse_args()
    
    # Initialize selected detectors
    detectors = []
    
    if "yolo" in args.detectors:
        try:
            detectors.append(YOLOWorldDetector())
        except Exception as e:
            print(f"Failed to initialize YOLOWorldDetector: {e}")
    
    if "detic" in args.detectors:
        try:
            detectors.append(DeticDetector())
        except Exception as e:
            print(f"Failed to initialize DeticDetector: {e}")
    
    if "owlvit" in args.detectors:
        try:
            detectors.append(OwlViTDetector())
        except Exception as e:
            print(f"Failed to initialize OwlViTDetector: {e}")
    
    if "sam_clip" in args.detectors:
        try:
            detectors.append(SamClipDetector())
        except Exception as e:
            print(f"Failed to initialize SamClipDetector: {e}")
    
    # Test each detector
    for detector in detectors:
        test_detector(detector, args.image, args.queries)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()