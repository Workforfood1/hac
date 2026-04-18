"""
Единая точка входа проекта.

Поддерживает два сценария:
- legacy: старая обработка одного видео через YOLO/OCR pipeline;
- hackathon: обработка видео 1/2/3 из задания хакатона и сравнение OCR-моделей.

По умолчанию `python main.py` запускает хакатонный сценарий для всех видео.
"""

import argparse
import io
import json
import logging
import os
import sys
import time
import warnings
import yaml
from pathlib import Path

from extract_scada import OCRAdapter, OUTPUT_DIR, process_video_full_ocr, process_video_smart


def configure_runtime():
    warnings.filterwarnings("ignore")
    logging.getLogger("easyocr").setLevel(logging.ERROR)
    os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')

    for stream_name in ('stdout', 'stderr'):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        encoding = (getattr(stream, 'encoding', '') or '').lower()
        if encoding == 'utf-8' or not hasattr(stream, 'buffer'):
            continue
        setattr(
            sys,
            stream_name,
            io.TextIOWrapper(stream.buffer, encoding='utf-8', errors='replace')
        )


def normalize_backend_name(name):
    normalized = (name or 'easyocr').strip().lower()
    if normalized == 'paddle':
        return 'paddlevl'
    if normalized == 'pytesseract':
        return 'tesseract'
    return normalized


def create_ocr_reader(backend):
    return OCRAdapter(normalize_backend_name(backend))


def summarize_region_rows(rows):
    frames = len(rows)
    non_empty = 0
    for row in rows:
        for key, value in row.items():
            if key.startswith('reg_') and value not in (None, ''):
                non_empty += 1
    average = non_empty / frames if frames else 0.0
    return frames, round(average, 3)


def run_hackathon_pipeline(videos=None, mode='smart', frame_step=None, max_frames=None, ocr_backend='easyocr'):
    videos = videos or [1, 2, 3]
    backend = normalize_backend_name(ocr_backend or os.environ.get('OCR_BACKEND', 'easyocr'))
    print("SCADA Video Extractor — Hackathon Mode")
    print(f"OCR backend: {backend}")
    print(f"Videos: {videos}")
    print(f"Mode: {mode}")
    print(f"Frame step: {frame_step or 'auto (1/sec)'}")
    print(f"Max frames: {max_frames or 'all'}")
    print()

    reader = create_ocr_reader(backend)
    summary = []

    for video_idx in videos:
        item = {'video': video_idx}
        try:
            if mode in ('smart', 'both'):
                rows = process_video_smart(
                    video_idx,
                    frame_step=frame_step,
                    max_frames=max_frames,
                    reader=reader,
                )
                item['smart_rows'] = len(rows)
            if mode in ('full', 'both'):
                rows = process_video_full_ocr(
                    video_idx,
                    frame_step=frame_step,
                    max_frames=max_frames,
                    reader=reader,
                )
                item['full_rows'] = len(rows)
        except Exception as exc:
            item['error'] = str(exc)
            print(f"  [FAIL] Видео {video_idx}: {exc}")
        summary.append(item)

    print("\n" + "=" * 60)
    print("DONE. Results:")
    for item in summary:
        details = []
        if 'smart_rows' in item:
            details.append(f"smart={item['smart_rows']}")
        if 'full_rows' in item:
            details.append(f"full={item['full_rows']}")
        if 'error' in item:
            details.append(f"error={item['error']}")
        print(f"  Видео {item['video']}: " + ", ".join(details))
    print(OUTPUT_DIR)

    return summary


def compare_video3_models(models=None, frame_step=None, max_frames=None, output_path=None):
    import openpyxl

    selected_models = [normalize_backend_name(model) for model in (models or ['easyocr', 'paddlevl'])]
    output = Path(output_path) if output_path else OUTPUT_DIR / 'ab_test_video3.xlsx'
    output.parent.mkdir(parents=True, exist_ok=True)

    print("A/B test for Видео 3")
    print(f"Models: {selected_models}")
    print(f"Frame step: {frame_step or 'auto (1/sec)'}")
    print(f"Max frames: {max_frames or 'all'}")

    results = []
    for model_name in selected_models:
        print(f"\nInitializing variant: {model_name}")
        try:
            reader = create_ocr_reader(model_name)
        except Exception as exc:
            error = str(exc)
            print(f"  Skipping {model_name}: {error}")
            results.append({
                'model': model_name,
                'frames': 0,
                'time_s': 0.0,
                'avg_values_per_frame': 0.0,
                'error': error,
            })
            continue

        start_time = time.time()
        try:
            rows = process_video_smart(3, frame_step=frame_step, max_frames=max_frames, reader=reader)
            frames, average = summarize_region_rows(rows)
            result = {
                'model': model_name,
                'frames': frames,
                'time_s': round(time.time() - start_time, 2),
                'avg_values_per_frame': average,
                'error': '',
            }
            print(f"Result: {result}")
            results.append(result)
        except Exception as exc:
            error = str(exc)
            print(f"  Run failed for {model_name}: {error}")
            results.append({
                'model': model_name,
                'frames': 0,
                'time_s': 0.0,
                'avg_values_per_frame': 0.0,
                'error': error,
            })

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(['model', 'frames', 'time_s', 'avg_values_per_frame', 'error'])
    for result in results:
        sheet.append([
            result['model'],
            result['frames'],
            result['time_s'],
            result['avg_values_per_frame'],
            result['error'],
        ])
    workbook.save(output)
    print(f"\nSaved A/B test results to {output}")

    return results, output


def build_cli_parser():
    parser = argparse.ArgumentParser(description='Unified entrypoint for the whole project')
    subparsers = parser.add_subparsers(dest='command')

    hackathon_parser = subparsers.add_parser('hackathon', help='Run the full hackathon pipeline for selected videos')
    hackathon_parser.add_argument('--videos', type=int, nargs='+', choices=[1, 2, 3], default=[1, 2, 3],
                                  help='Which videos to process')
    hackathon_parser.add_argument('--mode', choices=['smart', 'full', 'both'], default='smart',
                                  help='Extraction mode')
    hackathon_parser.add_argument('--frame_step', type=int, default=None,
                                  help='Frames to skip between readings (default: 1/sec based on FPS)')
    hackathon_parser.add_argument('--max_frames', type=int, default=None,
                                  help='Max frames per video (for testing)')
    hackathon_parser.add_argument('--ocr-backend', default=os.environ.get('OCR_BACKEND', 'easyocr'),
                                  choices=['easyocr', 'paddlevl', 'paddle', 'tesseract', 'pytesseract'],
                                  help='OCR backend for hackathon extraction')

    compare_parser = subparsers.add_parser('compare-video3', help='Compare OCR models on Видео 3')
    compare_parser.add_argument('--models', nargs='+', default=['easyocr', 'paddlevl'],
                                help='OCR models to compare')
    compare_parser.add_argument('--frame_step', type=int, default=None,
                                help='Frames to skip between readings (default: 1/sec based on FPS)')
    compare_parser.add_argument('--max_frames', type=int, default=None,
                                help='Max frames to process (for testing)')
    compare_parser.add_argument('--output', default=None, help='Path for the comparison Excel file')

    return parser


def main(argv=None):
    configure_runtime()
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv:
        run_hackathon_pipeline()
        return

    parser = build_cli_parser()
    args = parser.parse_args(argv)

    if args.command == 'hackathon':
        run_hackathon_pipeline(
            videos=args.videos,
            mode=args.mode,
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            ocr_backend=args.ocr_backend,
        )
        return

    if args.command == 'compare-video3':
        compare_video3_models(
            models=args.models,
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            output_path=args.output,
        )
        return

    parser.print_help()


if __name__ == '__main__':
    main()
