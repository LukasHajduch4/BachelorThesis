import cv2
import argparse
from detectors.yolo_world_detector import YOLOWorldDetector

def main(image_path):
    """
    Hlavný skript na testovanie detekcie objektov pomocou YOLO-World.
    """
    detector = YOLOWorldDetector()
    query_texts = ["pen", "sunglasses", "chair", "book", "niveacream"]
    detections, annotated_image = detector.detect(image_path, query_texts)
    
    print("Detekované objekty:")
    for obj in detections:
        print(f"{obj['label']} - Istota: {obj['score']:.2f}")
    
    cv2.imshow("Detekované objekty", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Cesta k obrázku")
    args = parser.parse_args()
    
    main(args.image)
