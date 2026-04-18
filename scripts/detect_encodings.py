"""Scan Хакатон folder for text/CSV files and detect likely encodings.

Usage:
    python scripts/detect_encodings.py
"""
from pathlib import Path
from encoding_utils import read_text, DEFAULT_ENCODING_ORDER


BASE = Path(r"c:\Users\kudzh_o\Documents\GitHub\hac\Хакатон «ИИ – АВТОМАТИЗАЦИЯ»")


def scan(folder: Path):
    files = list(folder.rglob('*.csv')) + list(folder.rglob('*.txt'))
    if not files:
        print("No .csv or .txt files found in", folder)
        return

    print(f"Scanning {len(files)} files under {folder}")
    for f in files:
        try:
            _, enc = read_text(str(f), encodings=DEFAULT_ENCODING_ORDER)
            print(f"{f.relative_to(folder)} -> {enc}")
        except Exception as e:
            print(f"{f.relative_to(folder)} -> ERROR: {e}")


if __name__ == '__main__':
    scan(BASE)
