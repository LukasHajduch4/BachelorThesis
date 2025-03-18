import cv2
import mediapipe as mp
import numpy as np

class HandPointingDetector:
    """
    Detects hands and pointing direction using MediaPipe.
    """
    
    def __init__(self):
        """
        Initialize the hand detector.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def detect_hands(self, image):
        """
        Detect hands in an image.
        
        Args:
            image: Input image (numpy.ndarray)
            
        Returns:
            list: List of detected hands
            numpy.ndarray: Annotated image
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        detected_hands = []
        annotated_image = image.copy()

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_type = results.multi_handedness[idx].classification[0].label  # Get Left/Right hand
                
                # Extract hand landmarks as a list of (x, y) coordinates
                hand_points = [
                    (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) 
                    for landmark in hand_landmarks.landmark
                ]

                detected_hands.append((hand_points, hand_type))  # Store with hand type

                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

        return detected_hands, annotated_image

    def detect_pointing_hand(self, detected_hands):
        """
        Determines which hand is actively pointing based on arm position and index finger extension.

        Args:
            detected_hands: List of tuples [(hand_points, hand_type)]

        Returns:
            tuple: (hand_points, hand_type) of the pointing hand, or None if no pointing hand detected.
        """
        if not detected_hands:
            return None  # No hands detected

        print(f"🔎 DEBUG: detected_hands = {detected_hands}")

        pointing_hand = None
        best_arm_position = float('inf')  # Lower y-value means hand is in front

        for hand_points, hand_type in detected_hands:
            print(f"🔎 DEBUG: Processing {hand_type} hand")

            if len(hand_points) < 21:
                continue  # Skip incomplete detections

            # Wrist and index finger landmarks
            WRIST = 0
            INDEX_FINGER_TIP = 8
            INDEX_FINGER_PIP = 6
            INDEX_FINGER_MCP = 5

            wrist = np.array(hand_points[WRIST])
            tip = np.array(hand_points[INDEX_FINGER_TIP])
            pip = np.array(hand_points[INDEX_FINGER_PIP])
            mcp = np.array(hand_points[INDEX_FINGER_MCP])

            # Measure how extended the index finger is
            finger_extension = (pip[1] - tip[1]) / (mcp[1] - pip[1] + 1e-6)  # Avoid division by zero

            # Use wrist y-coordinate to check which hand is more forward
            print(f"🟢 {hand_type} Wrist Position Y: {wrist[1]}, Best_arm_position: {best_arm_position}, Finger Extension Ratio: {finger_extension:.2f}")

            # Prioritize the hand that is in front (smaller y-value means closer to the camera)
            if wrist[1] < best_arm_position and finger_extension > 0.9:  # Allow slightly bent fingers
                best_arm_position = wrist[1]
                pointing_hand = (hand_points, hand_type)  # Select the forward hand with an extended finger

        if pointing_hand:
            print(f"✅ Pointing hand detected: {pointing_hand[1]}")
            return pointing_hand

        print("⚠ No pointing hand detected.")
        return None

        
    def detect_pointing_direction(self, hand_points, image_shape, objects_info):
        """
        Detects the pointing direction and extends the line until it touches a bounding box or reaches the image boundary.

        Args:
            hand_points: List of hand landmarks
            image_shape: Tuple (height, width) of the image
            objects_info: List of detected objects with 'box' info

        Returns:
            tuple: Start point and extended end point
        """
        if not hand_points or len(hand_points) < 21:
            return None

        INDEX_FINGER_TIP = 8
        INDEX_FINGER_PIP = 6

        index_tip = np.array(hand_points[INDEX_FINGER_TIP])
        index_pip = np.array(hand_points[INDEX_FINGER_PIP])

        direction_vector = index_tip - index_pip
        norm = np.linalg.norm(direction_vector)

        if norm == 0:
            return None

        normalized_direction = direction_vector / norm

        # Maximum extension distance (image boundaries)
        height, width = image_shape
        max_extension = max(height, width)  # Ensure it can extend across the image

        for scale in np.linspace(50, max_extension, num=200):  # Increase steps for finer control
            extended_point = index_tip + normalized_direction * scale

            # Check if the extended point goes out of bounds
            x, y = map(int, extended_point)
            if x < 0 or x >= width or y < 0 or y >= height:
                break  # Stop at image boundary

            # Check if the line intersects with any bounding box
            for obj in objects_info:
                x1, y1, x2, y2 = obj['box']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return index_pip, (x, y)  # Stop at the first detected object

        return index_pip, (x, y)  # If no object is found, stop at the image boundary
    
    def detect_pointing(self, image):
        """
        Detect pointing gestures in the image.
        
        Args:
            image: Input image
            
        Returns:
            tuple: Start and direction of pointing or None
            numpy.ndarray: Annotated image
        """
        detected_hands, annotated_image = self.detect_hands(image)
        
        if not detected_hands:
            return None, annotated_image
        
        pointing_info = self.detect_pointing_direction(detected_hands[0])
        
        if pointing_info:
            start_point, end_point = pointing_info
            cv2.line(annotated_image, tuple(map(int, start_point)), tuple(map(int, end_point)),
                     (255, 0, 0), 2)
        
        return pointing_info, annotated_image
