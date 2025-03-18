import cv2
import numpy as np
import argparse
from detectors.yolo_world_detector import YOLOWorldDetector
from pointing.hand_detector import HandPointingDetector
from pointing.pointing_analyzer import PointingAnalyzer

def test_pointing_system(image_path, query_texts=None):
    """
    Test YOLO-World object detection and pointing detection.

    Args:
        image_path: Path to the test image.
        query_texts: List of objects to detect (default: None for general objects).
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Initialize detectors
    object_detector = YOLOWorldDetector()
    hand_detector = HandPointingDetector()
    pointing_analyzer = PointingAnalyzer()

    # Step 1: Detect objects using YOLO-World
    print("\n🔹 Running YOLO-World Object Detection...")
    detected_objects, annotated_image = object_detector.detect(image, query_texts)
    print(f"✅ Detected {len(detected_objects)} objects.")

    # Step 2: Detect hands
    print("\n🔹 Running Hand Detection...")
    # Detect hands
    detected_hands, hand_annotated_image = hand_detector.detect_hands(image)

    if not detected_hands:
        print("❌ No hands detected.")
        pointing_direction = None
    else:
        # Determine which hand is actually pointing
        pointing_hand = hand_detector.detect_pointing_hand(detected_hands)

        if pointing_hand:
            active_hand, hand_type = pointing_hand
            print(f"🖐️ Active pointing hand: {hand_type}")

            # Get image shape
            image_shape = image.shape[:2]

            # Detect pointing direction for the selected hand
            pointing_direction = hand_detector.detect_pointing_direction(active_hand, image_shape, detected_objects)
        else:
            print("⚠ No pointing hand detected.")
            pointing_direction = None

    print(f"🔹1. Pointing Direction: {pointing_direction}")
    
    # Step 4: Determine which object the hand is pointing at
    pointed_object_idx = None
    if pointing_direction:
        pointed_object_idx = pointing_analyzer.find_pointed_object(pointing_direction, detected_objects)
    print(f"🔹2. Pointed Object Index: {pointed_object_idx}")
    
    # Step 5: Visualize the results
    final_image = pointing_analyzer.visualize_pointing_analysis(
        image, pointing_direction, detected_objects, pointed_object_idx
    )
    print("🔹3. Visualized Pointing Analysis.")

    # Display results
    cv2.imwrite("outputs/annotated_image.jpg", annotated_image)
    cv2.imwrite("outputs/hand_annotated_image.jpg", hand_annotated_image)
    cv2.imwrite("outputs/final_analysis.jpg", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Object and Hand Detection with Pointing Analysis")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--queries", type=str, nargs='+', default=["bottle", "cup", "chair", "book", "person"],
                        help="Query texts for YOLO detection")
    args = parser.parse_args()

    test_pointing_system(args.image, args.queries)
