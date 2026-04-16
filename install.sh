#!/bin/bash
# Скрипт для быстрой установки и запуска

echo "🚀 Установка task2-full..."

# Проверяем Python
if ! command -v python &> /dev/null; then
    echo "❌ Python не найден. Установите Python 3.8+"
    exit 1
fi

echo "✅ Python найден: $(python --version)"

# Создаем виртуальное окружение (опционально)
echo ""
echo "📦 Установка зависимостей..."
python -m pip install --upgrade pip

# Устанавливаем requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Зависимости установлены"
else
    echo "❌ requirements.txt не найден"
    exit 1
fi

echo ""
echo "✅ Установка завершена!"
echo ""
echo "🎥 Использование:"
echo "  python main.py video.mp4                    # Базовая обработка"
echo "  python main.py video.mp4 -o results.csv    # Экспорт в CSV"
echo "  python main.py video.mp4 -v                 # С визуализацией"
echo ""
echo "📚 Примеры в examples.py"
