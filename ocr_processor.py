import cv2
import numpy as np
import shutil
from pathlib import Path


class OCRProcessor:
    """Чтение текста и чисел с помощью PaddleOCR или Tesseract."""
    
    def __init__(self, use_paddle=True, lang="en"):
        self.use_paddle = use_paddle
        self.lang = lang
        self.active_lang = None
        self.ocr = None
        self.pytesseract = None
        self.ocr_available = True
        
        if use_paddle:
            try:
                from paddleocr import PaddleOCR

                # PaddleOCR принимает только один язык (строку), а не список.
                candidate_langs = self._normalize_languages(lang)
                init_error = None

                for candidate in candidate_langs:
                    try:
                        self.ocr = PaddleOCR(use_angle_cls=False, lang=candidate, show_log=False)
                        self.active_lang = candidate
                        break
                    except Exception as err:
                        init_error = err

                if self.ocr is None:
                    raise RuntimeError(
                        f"Не удалось инициализировать PaddleOCR ни для одного языка: {candidate_langs}"
                    ) from init_error

                print(f"✅ PaddleOCR инициализирована с языком: {self.active_lang}")
            except ImportError:
                print("⚠️ PaddleOCR не установлена. Используем Tesseract.")
                self.use_paddle = False
                try:
                    import pytesseract
                    self.pytesseract = pytesseract
                except ImportError:
                    raise ImportError("Ни PaddleOCR, ни Tesseract не установлены")
            except Exception as err:
                print(f"⚠️ Ошибка инициализации PaddleOCR: {err}. Используем Tesseract.")
                self.use_paddle = False
                try:
                    import pytesseract
                    self.pytesseract = pytesseract
                except ImportError:
                    raise ImportError("Ни PaddleOCR, ни Tesseract не установлены")
        else:
            try:
                import pytesseract
                self.pytesseract = pytesseract
                self._configure_tesseract_cmd()
            except ImportError:
                raise ImportError("Tesseract не установлен")

    def _configure_tesseract_cmd(self):
        """Находит tesseract.exe в PATH или в стандартных путях Windows."""
        if self.pytesseract is None:
            self.ocr_available = False
            return

        cmd = shutil.which("tesseract")
        if cmd:
            self.pytesseract.pytesseract.tesseract_cmd = cmd
            self.ocr_available = True
            return

        common_paths = [
            Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
            Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
        ]
        for candidate in common_paths:
            if candidate.exists():
                self.pytesseract.pytesseract.tesseract_cmd = str(candidate)
                self.ocr_available = True
                return

        self.ocr_available = False
        print("⚠️ Tesseract OCR не найден (нет tesseract.exe). OCR будет пропущен для этого запуска.")

    def _normalize_languages(self, lang):
        """Приводит входной язык(и) к списку кандидатов для PaddleOCR."""
        if isinstance(lang, (list, tuple, set)):
            langs = [str(item).strip() for item in lang if str(item).strip()]
        elif isinstance(lang, str):
            langs = [lang.strip()] if lang.strip() else []
        else:
            langs = []

        # Безопасный fallback, если конфиг пустой или некорректный.
        if not langs:
            langs = ["en"]

        # Убираем дубликаты, сохраняя порядок.
        unique_langs = list(dict.fromkeys(langs))
        return unique_langs
    
    def extract_text(self, frame, region=None):
        """
        Извлекает текст из кадра или региона.
        
        Args:
            frame: numpy array изображения
            region: tuple (x1, y1, x2, y2) для обработки региона
        
        Returns:
            dict с распознанным текстом и confidence
        """
        if region is not None:
            x1, y1, x2, y2 = region
            frame = frame[int(y1):int(y2), int(x1):int(x2)]
        
        if self.use_paddle and self.ocr is not None:
            results = self.ocr.ocr(frame, cls=False)
            
            text = ""
            confidences = []
            
            if results and results[0]:
                for line in results[0]:
                    # PaddleOCR обычно возвращает: [box, [text, score]]
                    if len(line) >= 2 and isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                        text += str(line[1][0]) + " "
                        confidences.append(float(line[1][1]))
            
            return {
                'text': text.strip(),
                'confidence': float(np.mean(confidences)) if confidences else 0.0,
                'raw': results
            }
        else:
            if not self.ocr_available or self.pytesseract is None:
                return {
                    'text': '',
                    'confidence': 0.0,
                    'raw': None
                }

            # Fallback на Tesseract
            try:
                text = self.pytesseract.image_to_string(frame)
            except Exception:
                # Не останавливаем весь пайплайн, если Tesseract недоступен/упал.
                return {
                    'text': '',
                    'confidence': 0.0,
                    'raw': None
                }
            return {
                'text': text.strip(),
                'confidence': 0.0,
                'raw': None
            }
    
    def extract_numbers(self, text):
        """Извлекает числа из текста."""
        import re
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return [float(n) if '.' in n else int(n) for n in numbers]
