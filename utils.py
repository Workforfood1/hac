import json
import csv
import cv2
import numpy as np
from pathlib import Path


class VideoReader:
    """Чтение видеофайла с оптимизацией для скорости."""
    
    def __init__(self, video_path, resize_scale=0.75, frame_skip=1):
        self.video_path = video_path
        self.resize_scale = resize_scale
        self.frame_skip = frame_skip
        
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_scale)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_scale)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = 0
    
    def read_frame(self):
        """Читает следующий кадр."""
        self.frame_count += 1
        ret, frame = self.cap.read()
        
        if ret and self.resize_scale != 1.0:
            frame = cv2.resize(frame, (self.width, self.height))
        
        return ret, frame
    
    def release(self):
        """Закрывает видеофайл."""
        self.cap.release()


class ResultFormatter:
    """Форматирование результатов в JSON/CSV."""
    
    @staticmethod
    def to_json(results, output_path=None):
        """Конвертирует результаты в JSON."""
        json_data = {
            'total_frames': len(results),
            'detections': results
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        return json_data
    
    @staticmethod
    def to_csv(results, output_path=None):
        """Конвертирует результаты в CSV."""
        if not results:
            return
        
        if output_path:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
    
    @staticmethod
    def to_text(results):
        """Конвертирует результаты в текстовый формат."""
        text = ""
        for i, result in enumerate(results):
            text += f"Frame {i+1}:\n"
            if isinstance(result, dict):
                for key, value in result.items():
                    text += f"  {key}: {value}\n"
            elif isinstance(result, list):
                for item in result:
                    text += f"  {item}\n"
            text += "\n"
        return text


class Timer:
    """Простой таймер для измерения производительности."""
    
    def __init__(self):
        import time
        self.time = time
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = self.time.time()
    
    def stop(self):
        self.end_time = self.time.time()
    
    def elapsed(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0
    
    def __str__(self):
        return f"{self.elapsed():.2f}s"


def draw_detections(frame, detections):
    """Рисует bounding boxes на кадре."""
    for det in detections:
        x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
        conf = det.get('confidence', 0)
        class_name = det.get('class_name', 'unknown')
        
        # Рисуем box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Рисуем label
        label = f"{class_name} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame
