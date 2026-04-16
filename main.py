"""
Полная архитектура для обработки видео:
- YOLOv8s: детекция зон
- PaddleOCR: чтение текста и чисел
- ORB/Homography: стабилизация видео
- Post-processing + Temporal fusion: финальная очистка

Оптимизировано для скорости: 1-2 секунды обработки.
"""

import argparse
import json
import sys
import yaml
from pathlib import Path

from video_stabilizer import VideoStabilizer
from zone_detector import ZoneDetector
from ocr_processor import OCRProcessor
from post_processor import PostProcessor
from utils import VideoReader, ResultFormatter, Timer, draw_detections


class VideoProcessor:
    """Главный класс для обработки видео."""
    
    def __init__(self, config_path='config.yaml'):
        # Загружаем конфиг
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print("🚀 Инициализация компонентов...")
        
        # Инициализируем stabilizer
        self.stabilizer = VideoStabilizer(
            smooth_window=self.config['stabilization'].get('smooth_window', 3)
        )
        
        # Инициализируем detector
        self.detector = ZoneDetector(
            model_name=self.config['model']['yolo_model'],
            conf_threshold=0.3
        )
        
        # Инициализируем OCR
        self.ocr = OCRProcessor(
            use_paddle=self.config['ocr'].get('use_paddle', True),
            lang=self.config['model']['ocr_lang']
        )
        
        # Инициализируем post-processor
        self.post_processor = PostProcessor(
            temporal_window=self.config['post_processing'].get('temporal_window', 5),
            conf_threshold=self.config['post_processing'].get('confidence_threshold', 0.5)
        )
        
        print("✅ Все компоненты инициализированы")
    
    def process_video(self, video_path, output_path=None, visualize=False):
        """
        Обрабатывает видеофайл.
        
        Args:
            video_path: путь к видео
            output_path: путь для сохранения результатов
            visualize: сохранять ли видео с визуализацией
        
        Returns:
            list: результаты для каждого кадра
        """
        timer = Timer()
        timer.start()
        
        print(f"\n📹 Открываем видео: {video_path}")
        
        if not Path(video_path).exists():
            print(f"❌ Файл не найден: {video_path}")
            return None
        
        # Открываем видео
        reader = VideoReader(
            video_path,
            resize_scale=self.config['video'].get('resize_scale', 0.75),
            frame_skip=self.config['video'].get('frame_skip', 1)
        )
        
        print(f"📊 Параметры: {reader.width}x{reader.height}, {reader.fps} FPS, {reader.total_frames} кадров")
        
        # Инициализируем writer для визуализации
        output_video = None
        if visualize:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video = cv2.VideoWriter(
                output_path.replace('.json', '_visual.mp4'),
                fourcc,
                reader.fps,
                (reader.width, reader.height)
            )
        
        all_results = []
        frame_num = 0
        
        print("\n⚡ Обработка видео...\n")
        
        while True:
            ret, frame = reader.read_frame()
            
            if not ret:
                break
            
            frame_num += 1
            
            # Шаг 1: Стабилизация
            stabilized = self.stabilizer.stabilize(frame)
            
            # Шаг 2: Детекция зон
            detections = self.detector.detect(stabilized)
            
            # Шаг 3: OCR для каждой зоны
            ocr_results = []
            for det in detections:
                region = (det['x1'], det['y1'], det['x2'], det['y2'])
                text = self.ocr.extract_text(stabilized, region=region)
                det['ocr_text'] = text['text']
                det['ocr_confidence'] = text['confidence']
                ocr_results.append(det)
            
            # Шаг 4: Post-processing
            processed = self.post_processor.apply_all(ocr_results)
            
            # Сохраняем результаты кадра
            frame_result = {
                'frame_num': frame_num,
                'detections': processed,
                'count': len(processed)
            }
            all_results.append(frame_result)
            
            # Визуализация (опционально)
            if output_video:
                vis_frame = draw_detections(stabilized.copy(), processed)
                output_video.write(vis_frame)
            
            # Прогресс
            if frame_num % max(1, reader.total_frames // 10) == 0:
                progress = (frame_num / reader.total_frames) * 100
                print(f"  Кадр {frame_num}/{reader.total_frames} ({progress:.1f}%)")
        
        reader.release()
        if output_video:
            output_video.release()
        
        timer.stop()
        
        print(f"\n✅ Обработка завершена за {timer}")
        print(f"   Обработано кадров: {frame_num}")
        print(f"   Среднее время на кадр: {timer.elapsed() / frame_num:.3f}s")
        
        # Сохраняем результаты
        if output_path:
            self._save_results(all_results, output_path)
        
        return all_results
    
    def _save_results(self, results, output_path):
        """Сохраняет результаты в файл."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Результаты сохранены: {output_path}")
        
        elif output_path.suffix == '.csv':
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Frame', 'Count', 'Text', 'Confidence'])
                for frame_result in results:
                    for det in frame_result['detections']:
                        writer.writerow([
                            frame_result['frame_num'],
                            frame_result['count'],
                            det.get('ocr_text', ''),
                            det.get('confidence', 0)
                        ])
            print(f"💾 Результаты сохранены: {output_path}")


# Импортируем cv2 здесь чтобы избежать циклических импортов
import cv2


def main():
    parser = argparse.ArgumentParser(description='Обработка видео: детекция + OCR + стабилизация')
    parser.add_argument('video', help='Путь к видеофайлу')
    parser.add_argument('-o', '--output', help='Путь для сохранения результатов (JSON или CSV)')
    parser.add_argument('-c', '--config', default='config.yaml', help='Путь к конфигурации')
    parser.add_argument('-v', '--visualize', action='store_true', help='Сохранять видео с визуализацией')
    
    args = parser.parse_args()
    
    # Проверяем что config существует
    if not Path(args.config).exists():
        print(f"❌ Конфиг не найден: {args.config}")
        print("📄 Используем конфиг по умолчанию")
        args.config = 'config.yaml'
    
    # Инициализируем процессор
    processor = VideoProcessor(args.config)
    
    # Обрабатываем видео
    results = processor.process_video(
        args.video,
        output_path=args.output or 'results.json',
        visualize=args.visualize
    )
    
    if results:
        print("\n📊 Сводка результатов:")
        total_detections = sum(r['count'] for r in results)
        print(f"   Всего детекций: {total_detections}")
        print(f"   Среднее на кадр: {total_detections / len(results):.1f}")


if __name__ == '__main__':
    main()
