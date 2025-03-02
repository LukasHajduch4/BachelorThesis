import cv2
import numpy as np
import json

class StereoProcessor:
    """
    Trieda pre spracovanie stereo obrazu a 3D rekonštrukciu.
    """
    
    def __init__(self, calibration_file):
        """
        Inicializácia stereo procesora s kalibračnými dátami.
        
        Args:
            calibration_file: Cesta k súboru s kalibračnými dátami
        """
        # Načítanie kalibračných dát
        with open(calibration_file, 'r') as f:
            calib_data = json.load(f)
        
        # Konverzia na numpy arrays
        self.K_left = np.array(calib_data['K_left'])
        self.D_left = np.array(calib_data['D_left'])
        self.K_right = np.array(calib_data['K_right'])
        self.D_right = np.array(calib_data['D_right'])
        self.R = np.array(calib_data['R'])
        self.T = np.array(calib_data['T'])
        
        # Výpočet mapy rektifikácie
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.K_left, self.D_left,
            self.K_right, self.D_right,
            (calib_data['width'], calib_data['height']),
            self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )
        
        # Vytvorenie mapy rektifikácie
        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(
            self.K_left, self.D_left, self.R1, self.P1,
            (calib_data['width'], calib_data['height']),
            cv2.CV_32FC1
        )
        
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(
            self.K_right, self.D_right, self.R2, self.P2,
            (calib_data['width'], calib_data['height']),
            cv2.CV_32FC1
        )
        
        # Vytvorenie stereo matchera
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16*16,
            blockSize=11,
            P1=8 * 3 * 11**2,
            P2=32 * 3 * 11**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )
    
    def rectify_images(self, left_image, right_image):
        """
        Rektifikácia stereo obrazov.
        
        Args:
            left_image: Ľavý obraz
            right_image: Pravý obraz
            
        Returns:
            tuple: Rektifikované ľavý a pravý obraz
        """
        left_rectified = cv2.remap(left_image, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_image, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR)
        
        return left_rectified, right_rectified
    
    def compute_disparity(self, left_rectified, right_rectified):
        """
        Výpočet mapy disparity.
        
        Args:
            left_rectified: Rektifikovaný ľavý obraz
            right_rectified: Rektifikovaný pravý obraz
            
        Returns:
            numpy.ndarray: Mapa disparity
        """
        # Konverzia na grayscale, ak je to potrebné
        if len(left_rectified.shape) == 3:
            left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_rectified
            right_gray = right_rectified
        
        # Výpočet disparity
        disparity = self.stereo_matcher.compute(left_gray, right_gray)
        
        # Normalizácia pre zobrazenie
        disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return disparity, disparity_normalized
    
    def reproject_to_3d(self, disparity):
        """
        Reprojekcia mapy disparity do 3D priestoru.
        
        Args:
            disparity: Mapa disparity
            
        Returns:
            numpy.ndarray: 3D bodové mračno (x, y, z)
        """
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
        return points_3d
    
    def get_3d_point_from_disparity(self, x, y, disparity):
        """
        Získanie 3D bodu z 2D súradníc a disparity.
        
        Args:
            x, y: 2D súradnice v ľavom obraze
            disparity: Mapa disparity
            
        Returns:
            tuple: 3D súradnice (X, Y, Z)
        """
        # Kontrola, či sú súradnice v platnom rozsahu
        if not (0 <= y < disparity.shape[0] and 0 <= x < disparity.shape[1]):
            return None
        
        # Získanie hodnoty disparity pre daný bod
        disp_value = disparity[y, x]
        
        # Reprojekcia na 3D bod
        if disp_value > 0:
            point_2d = np.array([[x, y, disp_value]], dtype=np.float32).reshape(1, 1, 3)
            point_3d = cv2.perspectiveTransform(point_2d, self.Q)
            return tuple(point_3d[0][0])
        
        return None
    
    def get_3d_point_from_pointing(self, pointing_direction, disparity):
        """
        Získanie 3D bodu v smere ukazovania.
        
        Args:
            pointing_direction: Tuple (start_point, end_point) definujúci smer ukazovania
            disparity: Mapa disparity
            
        Returns:
            tuple: 3D súradnice bodu, na ktorý sa ukazuje
        """
        if pointing_direction is None:
            return None
        
        start_point, end_point = pointing_direction
        
        # Vytvorenie parametrického vyjadrenia priamky
        line_vector = np.array(end_point) - np.array(start_point)
        line_direction = line_vector / np.linalg.norm(line_vector)
        
        # Prechádzanie bodov pozdĺž priamky a hľadanie prvého platného 3D bodu
        max_distance = 1000  # Maximálna vzdialenosť hľadania
        current_point = np.array(start_point)
        
        for distance in range(1, max_distance, 5):  # Kroky po 5 pixelov
            current_point = np.array(start_point) + distance * line_direction
            x, y = map(int, current_point)
            
            # Kontrola, či sme stále v obraze
            if not (0 <= y < disparity.shape[0] and 0 <= x < disparity.shape[1]):
                continue
            
            # Získanie 3D bodu
            point_3d = self.get_3d_point_from_disparity(x, y, disparity)
            
            if point_3d is not None and abs(point_3d[2]) < 10000:  # Kontrola, či je Z hodnota rozumná
                return point_3d
        
        return None