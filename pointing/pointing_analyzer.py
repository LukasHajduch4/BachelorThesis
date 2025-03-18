import numpy as np
import cv2

class PointingAnalyzer:
    """
    Analyzes pointing direction to identify the target object.
    """
    
    def __init__(self):
        pass
    
    def find_pointed_object(self, pointing_direction, objects_info):
        """
        Identifies the object being pointed at using closest distance.

        Args:
            pointing_direction: Tuple (start_point, end_point) defining pointing direction
            objects_info: List of detected objects

        Returns:
            int: Index of the pointed object or None
        """
        if pointing_direction is None or not objects_info:
            return None

        start_point, end_point = np.array(pointing_direction)
        line_vector = end_point - start_point
        norm = np.linalg.norm(line_vector)

        if norm == 0:
            return None

        line_direction = line_vector / norm

        best_object_idx = None
        min_distance = float('inf')

        for i, obj in enumerate(objects_info):
            x1, y1, x2, y2 = map(int, obj['box'])
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

            # Compute projection of the center onto the pointing vector
            projection = np.dot(center - start_point, line_direction) * line_direction + start_point
            distance = np.linalg.norm(projection - center)

            # Instead of only checking if inside the bounding box, allow a small margin (e.g., 50px)
            if distance < min_distance and distance < 50:  # 50px tolerance
                min_distance = distance
                best_object_idx = i

        return best_object_idx


    def visualize_pointing_analysis(self, image, pointing_direction, objects_info, pointed_object_idx=None):
        """
        Visualizes the pointing analysis and overlays debugging information.
        
        Args:
            image: Input image
            pointing_direction: (start_point, end_point) tuple defining pointing vector
            objects_info: List of detected objects with 'box', 'label', 'score'
            pointed_object_idx: Index of the pointed object (or None if no object detected)

        Returns:
            numpy.ndarray: Image with annotations for debugging
        """
        annotated_image = image.copy()

        if not objects_info:
            print("❌ No objects detected.")
            return annotated_image

        if pointing_direction:
            start_point, end_point = pointing_direction
            line_vector = np.array(end_point) - np.array(start_point)
            line_direction = line_vector / np.linalg.norm(line_vector)

            # Draw pointing vector
            cv2.line(annotated_image, tuple(map(int, start_point)), tuple(map(int, end_point)), (255, 0, 0), 2)
            cv2.circle(annotated_image, tuple(map(int, start_point)), 5, (0, 0, 255), -1)  # Start point in red

            print("\n🔎 Debugging Pointing Analysis:")
            print(f"Start Point: {start_point}")
            print(f"End Point: {end_point}")

        min_distance = float('inf')
        best_object_idx = None

        # Iterate over detected objects
        for i, obj in enumerate(objects_info):
            box = obj['box']
            label = obj['label']
            score = obj['score']
            
            x1, y1, x2, y2 = map(int, box)
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

            if pointing_direction:
                projection = np.dot(center - start_point, line_direction) * line_direction + start_point
                distance = np.linalg.norm(projection - center)

                print(f"🟢 {label} - Distance to Pointing Vector: {distance:.2f}")

                # Draw projection point
                cv2.circle(annotated_image, tuple(map(int, projection.astype(int))), 5, (0, 255, 255), -1)

                # Keep track of the closest object
                if distance < min_distance and distance < 50:  # Only select within 50-pixel threshold
                    min_distance = distance
                    best_object_idx = i

            # Draw bounding boxes
            color = (0, 255, 0) if i == best_object_idx else (255, 0, 0)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            label_text = f"{label}: {score:.2f}"
            if i == best_object_idx:
                label_text = "🎯 TARGET: " + label_text

            cv2.putText(annotated_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if best_object_idx is None:
            print("⚠ No object matched the pointing direction.")

        return annotated_image
