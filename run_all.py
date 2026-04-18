"""Compatibility wrapper around `main.py hackathon`."""

import argparse

from main import run_hackathon_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_step', type=int, default=None,
                        help='Frames to skip between readings (default: 1/sec based on FPS)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Max frames per video (for testing)')
    parser.add_argument('--videos', type=int, nargs='+', default=[1, 2, 3],
                        help='Which videos to process (default: 1 2 3)')
    parser.add_argument('--mode', choices=['smart', 'full', 'both'], default='smart',
                        help='Extraction mode')
    parser.add_argument('--ocr-backend', default='easyocr',
                        choices=['easyocr', 'paddlevl', 'paddle', 'tesseract', 'pytesseract'],
                        help='OCR backend')
    args = parser.parse_args()

    run_hackathon_pipeline(
        videos=args.videos,
        mode=args.mode,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        ocr_backend=args.ocr_backend,
    )


if __name__ == '__main__':
    main()
