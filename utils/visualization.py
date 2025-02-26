import cv2

def draw_detections(image, detections):
    """
    Vizualizácia detekovaných objektov.
    """
    for obj in detections:
        x1, y1, x2, y2 = map(int, obj['box'])
        label = f"{obj['label']} ({obj['score']:.2f})"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image