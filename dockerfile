# Dockerfile
FROM python:3.11-slim

# Устанавливаем системные зависимости для вашего приложения
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем всё приложение (кроме файлов из .dockerignore)
COPY . .

# Проверяем что скопировались важные файлы
RUN echo "=== Проверка файлов ===" && \
    ls -la && \
    echo "=== templates ===" && \
    ls -la templates/ && \
    echo "=== Python модули ===" && \
    python -c "import flask; print(f'Flask version: {flask.__version__}')"

# Открываем порт приложения
EXPOSE 5000

# Переменные окружения для Flask - ИСПРАВЛЕННЫЕ!
ENV PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    SECRET_KEY=github-actions-build-secret \
    DB_HOST=localhost \
    BOT_TOKEN=dummy_token_for_build \
    BOT_USERNAME=@crypto_app_bot

# Команда запуска приложения
CMD ["python", "app.py"]
