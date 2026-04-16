"""
Batch processing for all 3 SCADA videos.
Runs extraction on all 3 videos sequentially, saving results to Excel files.

Usage:
    python run_all.py                  # process all 3 videos, 1 frame/sec
    python run_all.py --frame_step 30  # every 30 frames
    python run_all.py --max_frames 50  # limit frames (for testing)
"""

import sys
import os
import io
import warnings
import logging
import argparse

warnings.filterwarnings("ignore")
logging.getLogger("easyocr").setLevel(logging.ERROR)
# Force UTF-8 output to avoid encoding errors on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

sys.path.insert(0, os.path.dirname(__file__))
from extract_scada import process_video_smart, load_easyocr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_step', type=int, default=None,
                        help='Frames to skip between readings (default: 1/sec based on FPS)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Max frames per video (for testing)')
    parser.add_argument('--videos', type=int, nargs='+', default=[1, 2, 3],
                        help='Which videos to process (default: 1 2 3)')
    args = parser.parse_args()

    print("SCADA Video Extractor — Batch Mode")
    print(f"Processing videos: {args.videos}")
    print(f"Frame step: {args.frame_step or 'auto (1/sec)'}")
    print(f"Max frames: {args.max_frames or 'all'}")
    print()

    # Load OCR once, reuse across videos
    print("Loading EasyOCR model (shared for all videos)...")
    reader = load_easyocr()
    print("OCR ready.\n")

    results = {}
    for idx in args.videos:
        try:
            rows = process_video_smart(
                idx,
                frame_step=args.frame_step,
                max_frames=args.max_frames,
                reader=reader,
            )
            results[idx] = len(rows)
            print(f"  [OK] Video {idx}: {len(rows)} rows extracted\n")
        except Exception as e:
            print(f"  [FAIL] Video {idx} FAILED: {e}\n")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("DONE. Results:")
    for idx, count in results.items():
        print(f"  Видео {idx}: {count} кадров → results/Результаты_смарт_{idx}.xlsx")
    print(f"\nFiles saved to: C:\\Users\\okudz\\Desktop\\хакатон\\task2-full\\results\\")


if __name__ == '__main__':
    main()
