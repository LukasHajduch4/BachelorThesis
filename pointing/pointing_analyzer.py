import numpy as np
import cv2

class PointingAnalyzer:
    """
    Trieda pre analýzu smeru ukazovania a identifikáciu objektov.
    """
    
    def __init__(self):
        """
        Inicializácia analyzátora.
        """
        pass
    
    def find_pointed_object(self, pointing_direction, objects_info, image_shape):
        """
        Nájdenie objektu, na ktorý sa ukazuje.
        
        Args:
            pointing_direction: Tuple (start_point, end_point) definujúci smer ukazovania
            objects_info: Slovník obsahujúci informácie o detekovaných objektoch
            image_shape: Tvar obrazu (výška, šírka)
            
        Returns:
            int: Index objektu, na ktorý sa ukazuje, alebo None ak žiadny
        """
        if pointing_direction is None or not objects_info['boxes']:
            return None
        
        start_point, end_point = pointing_direction
        
        # Vytvorenie parametrického vyjadrenia priamky
        line_vector = np.array(end_point) - np.array(start_point)
        line_magnitude = np.linalg.norm(line_vector)
        line_direction = line_vector / line_magnitude
        
        # Nájdenie priesečníkov s bounding boxami
        best_intersection = None
        best_distance = float('inf')
        best_object_idx = None
        
        for i, box in enumerate(objects_info['boxes']):
            x1, y1, x2, y2 = map(int, box)
            
            # Kontrola priečnika s hranicami bounding boxu
            intersections = []
            
            # Horizontálne hrany
            for y in [y1, y2]:
                # Parametre t pre y = start_point[1] + t * line_direction[1]
                if abs(line_direction[1]) > 1e-10:
                    t = (y - start_point[1]) / line_direction[1]
                    x = start_point[0] + t * line_direction[0]
                    
                    if t > 0 and x1 <= x <= x2:
                        intersections.append((x, y))
            
            # Vertikálne hrany
            for x in [x1, x2]:
                # Parametre t pre x = start_point[0] + t * line_direction[0]
                if abs(line_direction[0]) > 1e-10:
                    t = (x - start_point[0]) / line_direction[0]
                    y = start_point[1] + t * line_direction[1]
                    
                    if t > 0 and y1 <= y <= y2:
                        intersections.append((x, y))
            
            # Ak existujú priesečníky, nájdi najbližší
            if intersections:
                for intersection in intersections:
                    distance = np.linalg.norm(np.array(intersection) - np.array(start_point))
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_intersection = intersection
                        best_object_idx = i
        
        return best_object_idx
    
    def visualize_pointing_analysis(self, image, pointing_direction, objects_info, pointed_object_idx=None):
        """
        Vizualizácia analýzy ukazovania.
        
        Args:
            image: Vstupný obraz
            pointing_direction: Tuple (start_point, end_point) definujúci smer ukazovania
            objects_info: Slovník obsahujúci informácie o detekovaných objektoch
            pointed_object_idx: Index objektu, na ktorý sa ukazuje
            
        Returns:
            numpy.ndarray: Anotovaný obraz
        """
        annotated_image = image.copy()
        
        # Kreslenie bounding boxov všetkých objektov
        for i, (box, label, score) in enumerate(zip(objects_info['boxes'], 
                                                   objects_info['labels'], 
                                                   objects_info['scores'])):
            x1, y1, x2, y2 = map(int, box)
            
            # Farba boxu (zelená pre ukazovaný objekt, modrá pre ostatné)
            color = (0, 255, 0) if i == pointed_object_idx else (255, 0, 0)
            
            # Kreslenie bounding boxu
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Kreslenie textu
            label_text = f"{label}: {score:.2f}"
            if i == pointed_object_idx:
                label_text = "TARGET: " + label_text
                
            cv2.putText(annotated_image, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Kreslenie smeru ukazovania
        if pointing_direction:
            start_point, end_point = pointing_direction
            cv2.line(annotated_image, 
                     tuple(map(int, start_point)), 
                     tuple(map(int, end_point)),
                     (255, 0, 0), 2)
            
            # Kreslenie kruhu na začiatku smeru
            cv2.circle(annotated_image, tuple(map(int, start_point)), 5, (0, 0, 255), -1)
        
        return annotated_image