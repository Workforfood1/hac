"""Tune preprocessing for Paddle and then run comparison with larger frame count.

Usage:
  $env:PADDLE_DISABLE_ONEDNN='1'; $env:FLAGS_use_mkldnn='0'; .\.venv\Scripts\python.exe scripts\tune_and_run.py --run-frames 1000
"""
from pathlib import Path
import sys
import argparse
import cv2
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extract_scada import OCRAdapter

VIDEO_DIR = ROOT / 'Хакатон «ИИ – АВТОМАТИЗАЦИЯ»' / 'Задание 2' / 'Видео 3'
VIDEO_FILE = VIDEO_DIR / 'Видео 3.mp4'

parser = argparse.ArgumentParser()
parser.add_argument('--run-frames', type=int, default=1000)
parser.add_argument('--try-list', type=str, default='none,clahe,hist,sharpen,thres')
args = parser.parse_args()

TRY = args.try_list.split(',') if args.try_list else ['none']

# simple preprocessors
import numpy as np
import cv2

def preprocess_none(img):
    return img

def preprocess_clahe(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    g2 = clahe.apply(g)
    return cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR)

def preprocess_hist(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g2 = cv2.equalizeHist(g)
    return cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR)

def preprocess_sharpen(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

def preprocess_thres(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, t = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)

PRE_MAP = {
    'none': preprocess_none,
    'clahe': preprocess_clahe,
    'hist': preprocess_hist,
    'sharpen': preprocess_sharpen,
    'thres': preprocess_thres,
}

# load first frame
cap = cv2.VideoCapture(str(VIDEO_FILE))
ret, frame = cap.read()
cap.release()
if not ret:
    print('Could not read first frame')
    sys.exit(1)

num_re = re.compile(r"\d+[\.,]?\d*")

print('Tuning Paddle preprocessing on first frame...')
results = []
for name in TRY:
    pre = PRE_MAP.get(name, preprocess_none)
    img = pre(frame.copy())
    try:
        base = OCRAdapter('paddlevl')
        # wrap
        class Wrapped:
            def readtext(self, im, *a, **k):
                return base.readtext(pre(im), *a, **k)
        w = Wrapped()
        dets = w.readtext(img)
        # flatten detections to text
        texts = []
        for d in dets:
            if isinstance(d, dict):
                t = d.get('text') or d.get('transcription') or ''
            elif isinstance(d, (list, tuple)) and len(d) >= 2:
                t = d[1]
            else:
                t = str(d)
            texts.append(t)
        nums = [t for t in texts if num_re.search(t)]
        results.append((name, len(dets), len(nums)))
        print(f'{name}: dets={len(dets)} nums={len(nums)}')
    except Exception as e:
        print('error for', name, e)
        results.append((name, 0, 0))

# pick best by nums then dets
best = max(results, key=lambda x: (x[2], x[1]))
print('Best preprocess:', best)
chosen = best[0]

# run compare with chosen preprocess for paddlevl
print('Running comparison with paddle preprocess=', chosen)
cmd = f".venv\\Scripts\\python.exe scripts\\compare_paddle_easy_noise.py --max-frames {args.run_frames} --paddle-preprocess {chosen}"
print('Command:', cmd)
import subprocess
p = subprocess.run(cmd, shell=True)
print('compare exit', p.returncode)
