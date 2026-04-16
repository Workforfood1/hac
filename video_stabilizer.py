import cv2
import numpy as np
from collections import deque


class VideoStabilizer:
    """Быстрая стабилизация видео с помощью ORB и аффинных трансформаций."""
    
    def __init__(self, smooth_window=3):
        self.smooth_window = smooth_window
        self.orb = cv2.ORB_create(nfeatures=1500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.prev_gray = None
        self.prev_kp = None
        self.prev_desc = None
        self.transform_history = deque(maxlen=smooth_window)
    
    def stabilize(self, frame):
        """Стабилизирует один кадр."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_kp, self.prev_desc = self.orb.detectAndCompute(gray, None)
            return frame
        
        # Извлекаем ключевые точки текущего кадра
        curr_kp, curr_desc = self.orb.detectAndCompute(gray, None)
        
        if curr_desc is None or self.prev_desc is None:
            return frame
        
        # Сопоставляем ключевые точки
        matches = self.bf.match(self.prev_desc, curr_desc)
        
        if len(matches) < 4:
            return frame
        
        # Берем лучшие совпадения
        matches = sorted(matches, key=lambda x: x.distance)[:20]
        
        # Извлекаем координаты
        src_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in matches])
        
        try:
            # Вычисляем аффинную трансформацию
            M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, 
                                                   method=cv2.RANSAC, 
                                                   ransacReprojThreshold=4.0)
            if M is None:
                return frame
        except:
            return frame
        
        # Сглаживаем траекторию
        self.transform_history.append(M)
        M_smooth = np.mean(list(self.transform_history), axis=0)
        
        # Применяем трансформацию
        stabilized = cv2.warpAffine(frame, M_smooth, (w, h))
        
        # Обновляем состояние
        self.prev_gray = gray
        self.prev_kp = curr_kp
        self.prev_desc = curr_desc
        
        return stabilized
    
    def reset(self):
        """Сбрасывает состояние."""
        self.prev_gray = None
        self.prev_kp = None
        self.prev_desc = None
        self.transform_history.clear()
