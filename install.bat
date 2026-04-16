@echo off
REM Скрипт для быстрой установки и запуска на Windows

echo.
echo 🚀 Установка task2-full...
echo.

REM Проверяем Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python не найден. Установите Python 3.8+
    pause
    exit /b 1
)

echo ✅ Python найден:
python --version

REM Устанавливаем requirements
echo.
echo 📦 Установка зависимостей...
if exist requirements.txt (
    pip install -r requirements.txt
    echo ✅ Зависимости установлены
) else (
    echo ❌ requirements.txt не найден
    pause
    exit /b 1
)

echo.
echo ✅ Установка завершена!
echo.
echo 🎥 Использование:
echo   python main.py video.mp4                    # Базовая обработка
echo   python main.py video.mp4 -o results.csv    # Экспорт в CSV
echo   python main.py video.mp4 -v                 # С визуализацией
echo.
echo 📚 Примеры в examples.py
echo.
pause
