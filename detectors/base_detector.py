from abc import ABC, abstractmethod

class ObjectDetector(ABC):
    """
    Abstraktná trieda pre všetky detektory objektov.
    Definuje základné rozhranie, ktoré musia implementovať všetky detektory.
    """
    
    @abstractmethod
    def detect(self, image, query_texts=None, confidence_threshold=0.5):
        """
        Detekcia objektov v obraze.
        
        Args:
            image: Vstupný obraz (numpy.ndarray alebo cesta k súboru)
            query_texts: Voliteľný zoznam textových popisov pre open-vocabulary detekciu
            confidence_threshold: Prah istoty pre filtrovanie detekcií
            
        Returns:
            dict: Slovník obsahujúci detekované objekty s kľúčmi:
                - boxes: Zoznam bounding boxov [x1, y1, x2, y2]
                - labels: Zoznam názvov tried
                - scores: Zoznam hodnôt istoty
            numpy.ndarray: Anotovaný obraz s vyznačenými detekciami
        """
        pass
    
    @abstractmethod
    def get_model_info(self):
        """
        Získanie informácií o modeli.
        
        Returns:
            dict: Slovník obsahujúci informácie o modeli
        """
        pass