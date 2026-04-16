"""
Пример использования VideoProcessor с разными вариантами конфигурации.
"""

from main import VideoProcessor
import json
from pathlib import Path


def example_basic():
    """Базовая обработка с конфигом по умолчанию."""
    print("=" * 60)
    print("ПРИМЕР 1: Базовая обработка")
    print("=" * 60)
    
    processor = VideoProcessor('config.yaml')
    
    # Примечание: используйте реальный видеофайл
    video_path = 'test_video.mp4'
    
    if not Path(video_path).exists():
        print(f"⚠️  Видеофайл не найден: {video_path}")
        print("   Создайте тестовое видео или используйте существующее")
        return
    
    results = processor.process_video(
        video_path,
        output_path='results_basic.json'
    )
    
    # Вывод сводки
    if results:
        print("\n📊 Результаты:")
        for i, frame_result in enumerate(results[:3]):  # Первые 3 кадра
            print(f"\nКадр {frame_result['frame_num']}:")
            for det in frame_result['detections']:
                print(f"  - {det['class_name']}: {det.get('ocr_text', 'N/A')}")
                print(f"    Confidence: {det['confidence']:.2f}")


def example_fast():
    """Быстрая обработка с оптимизацией для скорости."""
    print("\n" + "=" * 60)
    print("ПРИМЕР 2: Быстрая обработка")
    print("=" * 60)
    
    # Создаем конфиг для быстрой обработки
    import yaml
    
    config = {
        'model': {'yolo_model': 'yolov8s.pt', 'ocr_lang': ['en', 'ru']},
        'video': {'frame_skip': 2, 'resize_scale': 0.5},  # Ускорение!
        'stabilization': {'smooth_window': 2},
        'ocr': {'use_paddle': True, 'confidence_threshold': 0.3},
        'post_processing': {'temporal_window': 3, 'confidence_threshold': 0.5},
        'output': {'format': 'json', 'include_confidence': True}
    }
    
    with open('config_fast.yaml', 'w') as f:
        yaml.dump(config, f)
    
    processor = VideoProcessor('config_fast.yaml')
    
    video_path = 'test_video.mp4'
    if not Path(video_path).exists():
        print(f"⚠️  Видеофайл не найден: {video_path}")
        return
    
    results = processor.process_video(
        video_path,
        output_path='results_fast.json'
    )
    
    print(f"\n⚡ Обработано быстро!")


def example_with_visualization():
    """Обработка с сохранением видео с визуализацией."""
    print("\n" + "=" * 60)
    print("ПРИМЕР 3: С визуализацией")
    print("=" * 60)
    
    processor = VideoProcessor('config.yaml')
    
    video_path = 'test_video.mp4'
    if not Path(video_path).exists():
        print(f"⚠️  Видеофайл не найден: {video_path}")
        return
    
    results = processor.process_video(
        video_path,
        output_path='results_visual.json',
        visualize=True  # Сохраняет видео с bounding boxes
    )
    
    print("\n🎥 Видео с визуализацией сохранено: results_visual_visual.mp4")


def compare_configs():
    """Сравнивает производительность разных конфигураций."""
    print("\n" + "=" * 60)
    print("ПРИМЕР 4: Сравнение конфигураций")
    print("=" * 60)
    
    import yaml
    import time
    
    configs = {
        'slow_but_accurate': {
            'video': {'frame_skip': 1, 'resize_scale': 1.0},
            'stabilization': {'smooth_window': 5},
            'post_processing': {'temporal_window': 7}
        },
        'balanced': {
            'video': {'frame_skip': 1, 'resize_scale': 0.75},
            'stabilization': {'smooth_window': 3},
            'post_processing': {'temporal_window': 5}
        },
        'fast': {
            'video': {'frame_skip': 2, 'resize_scale': 0.5},
            'stabilization': {'smooth_window': 2},
            'post_processing': {'temporal_window': 3}
        }
    }
    
    video_path = 'test_video.mp4'
    if not Path(video_path).exists():
        print(f"⚠️  Видеофайл не найден: {video_path}")
        return
    
    results_compare = {}
    
    for config_name, config_overrides in configs.items():
        print(f"\n🔄 Тестирование: {config_name}")
        
        # Создаем конфиг
        base_config = {
            'model': {'yolo_model': 'yolov8s.pt', 'ocr_lang': ['en', 'ru']},
            'video': {'frame_skip': 1, 'resize_scale': 0.75},
            'stabilization': {'smooth_window': 3},
            'ocr': {'use_paddle': True, 'confidence_threshold': 0.3},
            'post_processing': {'temporal_window': 5, 'confidence_threshold': 0.5},
            'output': {'format': 'json', 'include_confidence': True}
        }
        
        # Применяем переопределения
        for key, values in config_overrides.items():
            if key in base_config:
                base_config[key].update(values)
        
        with open(f'config_{config_name}.yaml', 'w') as f:
            yaml.dump(base_config, f)
        
        processor = VideoProcessor(f'config_{config_name}.yaml')
        
        start = time.time()
        results = processor.process_video(
            video_path,
            output_path=f'results_{config_name}.json'
        )
        elapsed = time.time() - start
        
        if results:
            total_dets = sum(r['count'] for r in results)
            results_compare[config_name] = {
                'time': elapsed,
                'detections': total_dets,
                'fps': len(results) / elapsed
            }
    
    # Вывод сравнения
    print("\n📊 Сравнение результатов:")
    print("-" * 60)
    print(f"{'Config':<20} {'Time (s)':<12} {'Detections':<12} {'FPS':<12}")
    print("-" * 60)
    
    for config_name, results in results_compare.items():
        print(f"{config_name:<20} {results['time']:<12.2f} {results['detections']:<12} {results['fps']:<12.2f}")


if __name__ == '__main__':
    print("\n🎥 Примеры использования VideoProcessor\n")
    
    # Раскомментируйте нужные примеры:
    
    # 1. Базовая обработка
    example_basic()
    
    # 2. Быстрая обработка
    # example_fast()
    
    # 3. С визуализацией
    # example_with_visualization()
    
    # 4. Сравнение конфигураций (требует времени)
    # compare_configs()
