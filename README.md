# Полная архитектура обработки видео (task2-full)

Система для быстрой обработки видео с детекцией, OCR и стабилизацией.

## Архитектура

```
Видео → Стабилизация (ORB) → Детекция (YOLOv8s) → OCR (PaddleOCR) → Post-processing → Результаты (JSON/CSV)
```

## Компоненты

1. **VideoStabilizer** (`video_stabilizer.py`)
   - Использует ORB для быстрого извлечения ключевых точек
   - Аффинные трансформации вместо гомографии
   - Скользящее окно сглаживания

2. **ZoneDetector** (`zone_detector.py`)
   - YOLOv8s для легкой и быстрой детекции
   - Конфигурируемый порог уверенности

3. **OCRProcessor** (`ocr_processor.py`)
   - PaddleOCR для многоязычного распознавания (EN + RU)
   - Извлечение текста и чисел из регионов

4. **PostProcessor** (`post_processor.py`)
   - Фильтрация по уверенности
   - Объединение перекрывающихся boxes (NMS)
   - Временная фильтрация (Temporal Fusion)

## Требования

```bash
pip install -r requirements.txt
```

## Использование

### Базовая обработка

```bash
python main.py input.mp4
```

Результаты сохранятся в `results.json`

### С опциями

```bash
python main.py input.mp4 -o results.csv -v --visualize
```

Параметры:
- `-o, --output`: путь для результатов (JSON или CSV)
- `-v, --visualize`: сохранять видео с визуализацией
- `-c, --config`: путь к файлу конфигурации

### Пример обработки из Python

```python
from main import VideoProcessor

processor = VideoProcessor('config.yaml')
results = processor.process_video('input.mp4', 'output.json', visualize=True)

for frame_result in results:
    print(f"Frame {frame_result['frame_num']}: {frame_result['count']} детекций")
    for det in frame_result['detections']:
        print(f"  - {det['class_name']}: {det['ocr_text']}")
```

## Конфигурация

Отредактируйте `config.yaml`:

```yaml
model:
  yolo_model: "yolov8s.pt"  # Используем s (small) для скорости
  
video:
  frame_skip: 1              # 1 = каждый кадр, 2 = каждый 2-й
  resize_scale: 0.75         # Уменьшение разрешения для скорости
  
stabilization:
  smooth_window: 3           # Больше = плавнее, но медленнее
  
post_processing:
  temporal_window: 5         # История для фильтрации
  confidence_threshold: 0.5  # Минимальная уверенность
```

## Производительность

На GPU (NVIDIA):
- Типичное время: 1-3 секунды на видео 30fps 1080p
- Зависит от: количества кадров, разрешения, количества объектов

Быстрые оптимизации:
- Увеличить `frame_skip` до 2-3
- Уменьшить `resize_scale` до 0.5
- Уменьшить `temporal_window` до 3

## Формат результатов

### JSON

```json
[
  {
    "frame_num": 1,
    "count": 2,
    "detections": [
      {
        "x1": 100.5, "y1": 200.3, "x2": 300.2, "y2": 400.1,
        "confidence": 0.95,
        "class": 0,
        "class_name": "person",
        "ocr_text": "12345",
        "ocr_confidence": 0.87
      }
    ]
  }
]
```

### CSV

| Frame | Count | Text | Confidence |
|-------|-------|------|------------|
| 1 | 2 | 12345 | 0.87 |
| 2 | 1 | 54321 | 0.92 |

## Структура проекта

```
task2-full/
├── main.py                # Точка входа
├── config.yaml            # Конфигурация
├── requirements.txt       # Зависимости
├── video_stabilizer.py    # Стабилизация
├── zone_detector.py       # Детекция YOLOv8s
├── ocr_processor.py       # OCR PaddleOCR
├── post_processor.py      # Post-processing
└── utils.py               # Утилиты
```

## Решение проблем

### Медленная обработка
- Увеличить `frame_skip` в config.yaml
- Уменьшить `resize_scale`
- Использовать GPU (CUDA)

### Низкая точность OCR
- Убедиться что `resize_scale` достаточно большой
- Увеличить `temporal_window` для большей фильтрации
- Проверить `confidence_threshold`

### Нестабильное видео
- Увеличить `smooth_window` в stabilization
- Проверить что видео не повреждено

## Лицензия

MIT
