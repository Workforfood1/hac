import numpy as np
from collections import deque


class PostProcessor:
    """Post-processing и temporal fusion результатов."""
    
    def __init__(self, temporal_window=5, conf_threshold=0.5):
        self.temporal_window = temporal_window
        self.conf_threshold = conf_threshold
        self.history = deque(maxlen=temporal_window)
    
    def filter_detections(self, detections, min_conf=0.3):
        """Фильтрует детекции по уверенности."""
        return [d for d in detections if d.get('confidence', 0) >= min_conf]
    
    def merge_overlapping_boxes(self, detections, iou_threshold=0.5):
        """Объединяет пересекающиеся bounding boxes."""
        if not detections:
            return []
        
        def iou(box1, box2):
            x1_1, y1_1, x2_1, y2_1 = box1['x1'], box1['y1'], box1['x2'], box1['y2']
            x1_2, y1_2, x2_2, y2_2 = box2['x1'], box2['y1'], box2['x2'], box2['y2']
            
            inter_x1 = max(x1_1, x1_2)
            inter_y1 = max(y1_1, y1_2)
            inter_x2 = min(x2_1, x2_2)
            inter_y2 = min(y2_1, y2_2)
            
            if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
                return 0.0
            
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = area1 + area2 - inter_area
            
            return inter_area / union_area if union_area > 0 else 0.0
        
        merged = []
        used = set()
        
        for i, det in enumerate(detections):
            if i in used:
                continue
            
            cluster = [det]
            used.add(i)
            
            for j in range(i + 1, len(detections)):
                if j not in used and iou(det, detections[j]) > iou_threshold:
                    cluster.append(detections[j])
                    used.add(j)
            
            # Объединяем cluster
            merged_box = {
                'x1': np.mean([d['x1'] for d in cluster]),
                'y1': np.mean([d['y1'] for d in cluster]),
                'x2': np.mean([d['x2'] for d in cluster]),
                'y2': np.mean([d['y2'] for d in cluster]),
                'confidence': np.mean([d['confidence'] for d in cluster]),
                'class': cluster[0]['class'],
                'class_name': cluster[0]['class_name']
            }
            merged.append(merged_box)
        
        return merged
    
    def temporal_fusion(self, current_detections):
        """
        Применяет временную фильтрацию для стабилизации результатов.
        """
        self.history.append(current_detections)
        
        if len(self.history) < 2:
            return current_detections
        
        # Усредняем по времени
        all_dets = []
        for frame_dets in self.history:
            all_dets.extend(frame_dets)
        
        if not all_dets:
            return current_detections
        
        # Группируем по классам
        by_class = {}
        for det in all_dets:
            cls = det['class']
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(det)
        
        # Усредняем по каждому классу
        fused = []
        for cls, dets in by_class.items():
            fused.append({
                'x1': np.mean([d['x1'] for d in dets]),
                'y1': np.mean([d['y1'] for d in dets]),
                'x2': np.mean([d['x2'] for d in dets]),
                'y2': np.mean([d['y2'] for d in dets]),
                'confidence': np.mean([d['confidence'] for d in dets]),
                'class': cls,
                'class_name': dets[0]['class_name']
            })
        
        return fused
    
    def apply_all(self, detections):
        """Применяет все стадии post-processing."""
        # 1. Фильтрация по уверенности
        detections = self.filter_detections(detections, self.conf_threshold)
        
        # 2. Объединение перекрывающихся boxes
        detections = self.merge_overlapping_boxes(detections)
        
        # 3. Временная фильтрация
        detections = self.temporal_fusion(detections)
        
        return detections
