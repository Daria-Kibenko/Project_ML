FROM python:3.11-slim

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Зависимости Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код проекта
COPY . .

# Создаём нужные папки
RUN mkdir -p logs ecopulse/db ecopulse/data ecopulse/model

# Открываем порты для API и дашборда
EXPOSE 8000 8501

# По умолчанию запускаем основной процесс
CMD ["python", "main.py"]
