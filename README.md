# HAC: извлечение SCADA-значений из видео

Проект для обработки видео из задания хакатона и сохранения результатов в Excel.

## Что сейчас поддерживается

- Единая точка входа: `main.py`
- Два режима CLI:
  - `hackathon` — обработка видео 1/2/3
  - `compare-video3` — сравнение OCR backend'ов на Видео 3
- OCR backend'ы:
  - `easyocr`
  - `paddlevl` (алиас: `paddle`)
  - `tesseract` (алиас: `pytesseract`)

> Legacy-пайплайн (YOLO/stabilizer/postprocessor) удалён из рабочего сценария.

## Установка

```bash
pip install -r requirements.txt
```

## Быстрый старт

### 1) Запуск всего хакатонного сценария

```bash
python main.py
```

Эквивалент:

```bash
python main.py hackathon --videos 1 2 3 --mode smart --ocr-backend easyocr
```

### 2) Запуск только одного видео

```bash
python main.py hackathon --videos 2 --mode smart --ocr-backend tesseract
```

### 3) Сравнение моделей на Видео 3

```bash
python main.py compare-video3 --models easyocr paddlevl tesseract
```

Результат сравнения: `results/ab_test_video3.xlsx`.

## Основные аргументы

### Команда `hackathon`

- `--videos 1 2 3` — какие видео обрабатывать
- `--mode smart|full|both` — режим извлечения
- `--frame_step N` — шаг по кадрам
- `--max_frames N` — ограничение числа обработанных кадров
- `--ocr-backend easyocr|paddlevl|paddle|tesseract|pytesseract`

### Команда `compare-video3`

- `--models ...` — список backend'ов для сравнения
- `--frame_step N`
- `--max_frames N`
- `--output PATH` — путь к Excel-файлу сравнения

## Где лежат результаты

- Smart-режим:
  - `results/Результаты_смарт_1.xlsx`
  - `results/Результаты_смарт_2.xlsx`
  - `results/Результаты_смарт_3.xlsx`
- Full OCR:
  - `results/Результаты_fullOCR_*.xlsx`
- Сравнение моделей:
  - `results/ab_test_video3.xlsx`

### Важное обновление

В smart-результаты добавлена колонка времени обработки каждого кадра:

- `Обработка кадра, сек`

Это позволяет делать пер-кадровый замер производительности прямо в итоговой таблице.

## Tesseract (Windows)

Если используется `--ocr-backend tesseract`, должен быть установлен бинарник Tesseract OCR.

Автопоиск выполняется в стандартных путях, включая:

- `C:\Program Files\Tesseract-OCR\tesseract.exe`
- `%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe`

Можно явно задать путь через переменную окружения:

```powershell
$env:TESSERACT_CMD = "C:\path\to\tesseract.exe"
python main.py hackathon --videos 3 --ocr-backend tesseract
```

## Структура (актуально)

- `main.py` — CLI и orchestration
- `extract_scada.py` — логика извлечения, OCR adapter, сохранение Excel
- `run_all.py` — совместимый wrapper запуска
- `scripts/` — вспомогательные утилиты
- `results/` — выходные файлы

## Примечания по стабильности

- На больших кадрах `easyocr` может быть заметно медленнее и нестабильнее.
- Для длинных прогонов на CPU часто практичнее `tesseract`.
