"""Convert CSV files under the Хакатон folder to UTF-8 (with BOM) in-place.

Backs up original files with a .bak extension.
"""
from pathlib import Path
from encoding_utils import read_text


BASE = Path(r"c:\Users\kudzh_o\Documents\GitHub\hac\Хакатон «ИИ – АВТОМАТИЗАЦИЯ»")


def convert_file(path: Path, target_enc='utf-8-sig'):
    print(f"Converting: {path}")
    text, enc = read_text(str(path))
    if enc.lower() in ('utf-8', 'utf8'):
        print(f"  Already UTF-8 -> skipping")
        return False

    # Backup original
    bak = path.with_suffix(path.suffix + '.bak')
    if not bak.exists():
        path.rename(bak)
        bak.write_bytes(bak.read_bytes())
        # above line ensures content remains; actually bak already contains data after rename

    # Write converted file
    with open(path, 'w', encoding=target_enc, newline='') as f:
        f.write(text)
    print(f"  {path.name}: {enc} -> {target_enc}")
    return True


def scan_and_convert(folder: Path):
    files = list(folder.rglob('*.csv'))
    if not files:
        print("No CSV files found to convert.")
        return []

    converted = []
    for f in files:
        try:
            text, enc = read_text(str(f))
            if enc.lower() != 'utf-8':
                # perform conversion (create backup)
                # Use safe write: rename original to .bak then write
                bak = f.with_suffix(f.suffix + '.bak')
                if not bak.exists():
                    f.rename(bak)
                    # write new utf-8 file from decoded text
                    with open(f, 'w', encoding='utf-8-sig', newline='') as out:
                        out.write(text)
                    converted.append((f, enc))
                    print(f"Converted: {f.relative_to(folder)} ({enc} -> utf-8-sig)")
                else:
                    print(f"Backup already exists for {f}, skipping to avoid overwrite")
        except Exception as e:
            print(f"Error converting {f}: {e}")

    return converted


if __name__ == '__main__':
    conv = scan_and_convert(BASE)
    print(f"Done. Converted {len(conv)} files.")
