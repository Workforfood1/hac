import torch
import numpy as np


class ZoneDetector:
    """Детекция зон с помощью YOLOv8s."""
    
    def __init__(self, model_name="yolov8s.pt", conf_threshold=0.5):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics не установлена. Выполните: pip install ultralytics")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_name)
        self.model.to(self.device)
        self.conf_threshold = conf_threshold
    
    def detect(self, frame):
        """
        Детектирует объекты/зоны в кадре.
        
        Returns:
            list: список детекций [x1, y1, x2, y2, conf, class_id, class_name]
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    class_name = result.names[int(cls)]
                    detections.append({
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2),
                        'confidence': float(conf),
                        'class': int(cls),
                        'class_name': class_name
                    })
        
        return detections
