# 🌿 EcoPulse - MVP. Домашнее задание №8

**Проект:** EcoPulse - Система мониторинга ESG-репутации  
**Стек:** Python · FastAPI · SQLite · Streamlit · Docker · GigaChat

---

## 1. Доменная модель сервиса

```
┌─────────────────────────────────────────────────────────┐
│                    ДОМЕННАЯ МОДЕЛЬ                       │
├──────────────┬──────────────┬──────────────┬────────────┤
│    Post      │   Feedback   │LabelingQueue │ RetainLog  │
├──────────────┼──────────────┼──────────────┼────────────┤
│ id           │ id           │ id           │ id         │
│ channel      │ post_id (FK) │ post_id (FK) │triggered_at│
│ message_id   │ user_id      │ uncertainty  │n_new_samples│
│ text         │ is_correct   │ status       │ f1_before  │
│ published    │ created_at   │ manual_label │ f1_after   │
│ parsed_at    └──────────────┴──────────────┴────────────┘
│ model_score
│ source_weight
│ final_score
│ label (0/1)
│ notified
└──────────────

Бизнес-правила:
• Post.final_score = model_score × 0.7 + source_weight × 0.3
• label=1 (critical) если final_score >= THRESHOLD (0.65)
• Если model_score ∈ [0.40, 0.60] → пост попадает в LabelingQueue
• Feedback от аналитика → дообучение модели при накоплении ≥200 меток
```

---

## 2. Хранение данных (СУБД)

Используется **SQLite** - встроенная БД, не требует отдельного сервера.  
Путь: `ecopulse/db/ecopulse.db`

Инициализация:
```bash
python -c "from ecopulse.db.storage import init_db; init_db()"
```

Схема таблиц описана в `ecopulse/db/storage.py` → функция `init_db()`.

---

## 3. REST API (FastAPI)

Файл: `ecopulse/api/main.py`

```bash
# Запуск
uvicorn ecopulse.api.main:app --host 0.0.0.0 --port 8000 --reload

# Документация
http://localhost:8000/docs
```

### Эндпоинты:

| Метод | URL | Описание |
|-------|-----|---------|
| GET | `/health` | Статус сервиса |
| GET | `/posts` | Список постов с фильтрами |
| GET | `/posts/critical` | Только критические посты |
| POST | `/posts/classify` | Классифицировать текст |
| POST | `/posts/forecast` | Прогноз реакции через GigaChat |
| GET | `/stats` | Статистика системы |
| POST | `/feedback` | Сохранить фидбек 👍/👎 |
| GET | `/labeling/queue` | Очередь для Active Learning |
| POST | `/labeling/{id}` | Разметить пост |

---

## 4. Пользовательский интерфейс (Streamlit)

Файл: `ecopulse/dashboard/app.py`

```bash
python -m streamlit run ecopulse/dashboard/app.py
# Открыть: http://localhost:8501
```

### Вкладки:
- **📊 Обзор** - KPI-метрики, стековый bar по дням, таблица постов
- **📈 Аналитика** - пончик источников, violin скоров, тепловая карта, scatter, тренд
- **🔴 Критические** - gauge доли + список инцидентов
- **🏷 Разметка** - Active Learning: разметка неопределённых постов
- **🧠 GigaChat** - прогноз реакции с вероятными комментариями и рекомендацией

---

## 5. Тесты

Файл: `tests/test_ecopulse.py`

```bash
pip install pytest pytest-asyncio httpx
pytest tests/ -v
```

Покрытие критических частей:
- `test_storage` - сохранение/чтение из БД
- `test_predictor` - классификация текста
- `test_api_health` - доступность REST API
- `test_api_classify` - эндпоинт классификации
- `test_keywords` - фильтрация стоп-слов и приоритетных слов
- `test_feedback` - сохранение фидбека

---

## 6. Docker

```bash
# Сборка и запуск
docker-compose up --build

# Только сборка
docker build -t ecopulse .

# Фоновый режим
docker-compose up -d --build
```

Сервисы:
- `ecopulse` - парсер + Telegram-бот (порт не нужен)
- `dashboard` - Streamlit UI → http://localhost:8501
- `api` - FastAPI REST → http://localhost:8000

---

## 7. Масштабирование воркеров с моделью

```bash
# Запуск 3 воркеров классификации параллельно
docker-compose up --scale worker=3

# Проверка
docker-compose ps
```

Воркеры читают посты из общей очереди (SQLite) и классифицируют независимо.  
ONNX-модель загружается в каждом воркере - инференс полностью stateless.

---

## Структура проекта

```
ecopulse_project/
├── main.py                    # точка входа
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── requirements.txt
├── tests/
│   └── test_ecopulse.py
└── ecopulse/
    ├── __init__.py
    ├── config.py
    ├── api/
    │   ├── __init__.py
    │   └── main.py            # FastAPI REST
    ├── bot/
    │   ├── bot.py
    │   ├── digest.py
    │   └── retrain_trigger.py
    ├── dashboard/
    │   └── app.py             # Streamlit UI
    ├── data/
    │   ├── collect.py
    │   ├── label.py
    │   └── preprocess.py
    ├── db/
    │   └── storage.py
    ├── inference/
    │   ├── predictor.py
    │   ├── ner.py
    │   └── trend_detector.py
    ├── model/
    │   ├── train.py
    │   ├── evaluate.py
    │   └── export.py
    ├── monitoring/
    │   └── drift.py
    └── parser/
        ├── keywords.py
        ├── runner.py
        └── sources.py
```
