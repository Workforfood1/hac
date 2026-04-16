"""
Быстрый старт - минимальный пример использования
"""

from pathlib import Path
from main import VideoProcessor


def quick_start(video_path='input.mp4'):
    """Самый простой способ обработать видео."""
    
    # Проверяем видео
    if not Path(video_path).exists():
        print(f"❌ Видео не найдено: {video_path}")
        print("\n💡 Используйте свой видеофайл: python quick_start.py path/to/video.mp4")
        return
    
    # Инициализируем процессор
    print("🚀 Инициализация процессора...")
    processor = VideoProcessor('config.yaml')
    
    # Обрабатываем видео
    print(f"\n🎥 Обработка: {video_path}")
    results = processor.process_video(
        video_path,
        output_path='results.json'
    )
    
    # Выводим результаты
    if results:
        print("\n📊 РЕЗУЛЬТАТЫ:")
        print("=" * 60)
        
        total_detections = 0
        for frame_result in results[:5]:  # Первые 5 кадров
            if frame_result['detections']:
                print(f"\nКадр №{frame_result['frame_num']}:")
                for det in frame_result['detections']:
                    print(f"  ✓ {det['class_name']}")
                    if det.get('ocr_text'):
                        print(f"    Текст: {det['ocr_text']}")
                    print(f"    Уверенность: {det['confidence']:.1%}")
                    total_detections += 1
        
        print("\n" + "=" * 60)
        print(f"📈 Всего детекций в видео: {sum(r['count'] for r in results)}")
        print(f"💾 Результаты сохранены: results.json")


if __name__ == '__main__':
    import sys
    
    # Получаем путь к видео из аргументов
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = 'input.mp4'
    
    quick_start(video_path)
