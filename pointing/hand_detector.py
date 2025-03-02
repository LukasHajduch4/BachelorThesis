import cv2
import mediapipe as mp
import numpy as np

class HandPointingDetector:
    """
    Detektor ruky a smeru ukazovania pomocou MediaPipe.
    """
    
    def __init__(self):
        """
        Inicializácia detektora ruky.
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
        Detekcia rúk v obraze.
        
        Args:
            image: Vstupný obraz (numpy.ndarray)
            
        Returns:
            list: Zoznam detekovaných rúk
            numpy.ndarray: Anotovaný obraz
        """
        # Konverzia na RGB, ak je to potrebné
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image.dtype == np.uint8:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Detekcia rúk
        results = self.hands.process(rgb_image)
        
        # Príprava výstupu
        detected_hands = []
        annotated_image = image.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Uloženie bodov ruky
                hand_points = []
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                    hand_points.append((x, y))
                
                detected_hands.append(hand_points)
                
                # Kreslenie bodov ruky
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        
        return detected_hands, annotated_image
    
    def detect_pointing_direction(self, hand_points):
        """
        Detekcia smeru ukazovania pre jednu ruku.
        
        Args:
            hand_points: Zoznam bodov ruky
            
        Returns:
            tuple: Bod začiatku a smeru ukazovania, alebo None ak nie je detekované
        """
        if not hand_points or len(hand_points) < 21:
            return None
        
        # Indexy pre kľúčové body ruky v MediaPipe
        WRIST = 0
        INDEX_FINGER_TIP = 8
        INDEX_FINGER_PIP = 6
        INDEX_FINGER_MCP = 5
        
        # Získanie bodov
        wrist = np.array(hand_points[WRIST])
        index_tip = np.array(hand_points[INDEX_FINGER_TIP])
        index_pip = np.array(hand_points[INDEX_FINGER_PIP])
        index_mcp = np.array(hand_points[INDEX_FINGER_MCP])
        
        # Výpočet smeru ukazovania (z MCP cez PIP až k špičke prsta)
        direction_vector = index_tip - index_mcp
        
        # Normalizácia vektora
        norm = np.linalg.norm(direction_vector)
        if norm == 0:
            return None
        
        normalized_direction = direction_vector / norm
        
        # Predĺženie vektora pre vizualizáciu
        extended_point = index_tip + normalized_direction * 200
        
        return index_mcp, tuple(map(int, extended_point))
    
    def detect_pointing(self, image):
        """
        Detekcia smeru ukazovania v obraze.
        
        Args:
            image: Vstupný obraz
            
        Returns:
            tuple: Bod začiatku a smeru ukazovania, alebo None ak nie je detekované
            numpy.ndarray: Anotovaný obraz
        """
        # Detekcia rúk
        detected_hands, annotated_image = self.detect_hands(image)
        
        # Ak neboli detekované ruky, vrátime None
        if not detected_hands:
            return None, annotated_image
        
        # Pre jednoduchosť berieme len prvú detekovanú ruku
        pointing_info = self.detect_pointing_direction(detected_hands[0])
        
        # Ak bol detekovaný smer ukazovania, nakreslíme ho
        if pointing_info:
            start_point, end_point = pointing_info
            cv2.line(annotated_image, 
                     tuple(map(int, start_point)), 
                     tuple(map(int, end_point)),
                     (255, 0, 0), 2)
        
        return pointing_info, annotated_image