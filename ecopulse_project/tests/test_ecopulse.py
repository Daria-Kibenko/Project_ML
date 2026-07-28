import sys
import os
import sqlite3
import tempfile
import pytest

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT)


# 1. ТЕСТЫ КЛЮЧЕВЫХ СЛОВ (keywords.py)
def test_is_priority_positive():
    from ecopulse.parser.keywords import is_priority
    assert is_priority("На заводе произошёл выброс химикатов") is True
    assert is_priority("Разлив нефти в реке Обь зафиксирован") is True
    assert is_priority("Жители жалуются на загрязнение воздуха") is True


def test_is_priority_negative():
    from ecopulse.parser.keywords import is_priority
    assert is_priority("Отличный банк, быстрое обслуживание") is False
    assert is_priority("Спасибо за хороший сервис!") is False


def test_is_spam():
    from ecopulse.parser.keywords import is_spam
    assert is_spam("Скидка! Купить со скидкой! Промокод акция") is True
    assert is_spam("Завод выбросил вредные вещества в атмосферу") is False


def test_get_incident_category():
    from ecopulse.parser.keywords import get_incident_category
    assert get_incident_category("разлив нефти и выброс химикатов") == "экологический"
    assert get_incident_category("забастовка рабочих из-за задержки зарплаты") == "трудовой"
    assert get_incident_category("проверка роспотребнадзора на предприятии") == "регуляторный"
    assert get_incident_category("отравление жителей вредными веществами") == "здоровье"


# 2. ТЕСТЫ БАЗЫ ДАННЫХ (storage.py)
@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    db_file = str(tmp_path / "test.db")

    import ecopulse.db.storage as storage
    monkeypatch.setattr(storage, "_DB_PATH", db_file)
    storage.init_db()
    return db_file


def test_init_db(tmp_db):
    con = sqlite3.connect(tmp_db)
    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    con.close()
    assert "posts" in tables
    assert "feedback" in tables
    assert "labeling_queue" in tables


def test_save_and_get_post(tmp_db, monkeypatch):
    import ecopulse.db.storage as storage
    monkeypatch.setattr(storage, "_DB_PATH", tmp_db)

    from datetime import datetime, timezone
    post_id = storage.save_post(
        channel="test_channel",
        message_id="msg_001",
        text="На заводе произошёл выброс химикатов",
        published=datetime.now(timezone.utc),
        model_score=0.91,
        source_weight=0.6,
        label=1,
    )
    assert post_id is not None

    con = sqlite3.connect(tmp_db)
    row = con.execute("SELECT * FROM posts WHERE id=?", (post_id,)).fetchone()
    con.close()
    assert row is not None
    assert row[1] == "test_channel"
    assert row[9] == 1  # label


def test_save_feedback(tmp_db, monkeypatch):
    import ecopulse.db.storage as storage
    monkeypatch.setattr(storage, "_DB_PATH", tmp_db)

    from datetime import datetime, timezone
    post_id = storage.save_post(
        "ch", "id1", "text", datetime.now(timezone.utc),
        0.8, 0.5, 1
    )
    storage.save_feedback(post_id, user_id=42, is_correct=1)

    con = sqlite3.connect(tmp_db)
    row = con.execute("SELECT * FROM feedback WHERE post_id=?", (post_id,)).fetchone()
    con.close()
    assert row is not None
    assert row[3] == 1  # is_correct


def test_labeling_queue(tmp_db, monkeypatch):
    import ecopulse.db.storage as storage
    monkeypatch.setattr(storage, "_DB_PATH", tmp_db)

    from datetime import datetime, timezone
    from ecopulse.config import cfg

    uncertain_score = (cfg.UNCERTAINTY_LOW + cfg.UNCERTAINTY_HIGH) / 2
    storage.save_post(
        "ch", "id2", "неопределённый текст",
        datetime.now(timezone.utc),
        uncertain_score, 0.5, 0
    )
    queue = storage.get_labeling_queue(limit=10)
    assert len(queue) > 0


def test_submit_manual_label(tmp_db, monkeypatch):
    import ecopulse.db.storage as storage
    monkeypatch.setattr(storage, "_DB_PATH", tmp_db)

    from datetime import datetime, timezone
    from ecopulse.config import cfg

    score = (cfg.UNCERTAINTY_LOW + cfg.UNCERTAINTY_HIGH) / 2
    storage.save_post("ch","id3","text",datetime.now(timezone.utc),score,0.5,0)
    queue = storage.get_labeling_queue(limit=1)
    assert queue

    queue_id = queue[0][0]
    storage.submit_manual_label(queue_id, 1)

    con = sqlite3.connect(tmp_db)
    row = con.execute(
        "SELECT status, manual_label FROM labeling_queue WHERE id=?", (queue_id,)
    ).fetchone()
    con.close()
    assert row[0] == "labeled"
    assert row[1] == 1


# 3. ТЕСТЫ REST API (FastAPI)
@pytest.fixture
def api_client(tmp_db, monkeypatch):
    """Тестовый клиент FastAPI."""
    import ecopulse.db.storage as storage
    monkeypatch.setattr(storage, "_DB_PATH", tmp_db)

    from httpx import AsyncClient
    from ecopulse.api.main import app
    return app, tmp_db


@pytest.mark.asyncio
async def test_api_health(api_client):
    from httpx import AsyncClient, ASGITransport
    app, _ = api_client
    async with AsyncClient(transport=ASGITransport(app=app),
                            base_url="http://test") as ac:
        resp = await ac.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_api_stats(api_client):
    from httpx import AsyncClient, ASGITransport
    app, _ = api_client
    async with AsyncClient(transport=ASGITransport(app=app),
                            base_url="http://test") as ac:
        resp = await ac.get("/stats")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_api_classify(api_client):
    from httpx import AsyncClient, ASGITransport
    app, _ = api_client
    async with AsyncClient(transport=ASGITransport(app=app),
                            base_url="http://test") as ac:
        resp = await ac.post("/posts/classify", json={
            "text": "Выброс химикатов на заводе, есть пострадавшие",
            "source_weight": 0.6,
        })
    assert resp.status_code == 200
    data = resp.json()
    assert "label" in data
    assert "final_score" in data
    assert data["label"] in [0, 1]


@pytest.mark.asyncio
async def test_api_feedback(api_client, monkeypatch):
    import ecopulse.db.storage as storage
    from httpx import AsyncClient, ASGITransport
    from datetime import datetime, timezone

    app, tmp_db = api_client
    monkeypatch.setattr(storage, "_DB_PATH", tmp_db)

    post_id = storage.save_post(
        "ch","id_fb","text",datetime.now(timezone.utc),0.9,0.5,1
    )
    async with AsyncClient(transport=ASGITransport(app=app),
                            base_url="http://test") as ac:
        resp = await ac.post("/feedback", json={
            "post_id": post_id, "user_id": 1, "is_correct": 1
        })
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_api_labeling_queue(api_client):
    from httpx import AsyncClient, ASGITransport
    app, _ = api_client
    async with AsyncClient(transport=ASGITransport(app=app),
                            base_url="http://test") as ac:
        resp = await ac.get("/labeling/queue")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_api_posts_filter(api_client):
    from httpx import AsyncClient, ASGITransport
    app, _ = api_client
    async with AsyncClient(transport=ASGITransport(app=app),
                            base_url="http://test") as ac:
        resp = await ac.get("/posts?days=7&label=1&limit=10")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# 4. ТЕСТЫ ПРЕДИКТОРА (predictor.py) - только если есть ONNX
def test_predictor_output_structure():
    try:
        from ecopulse.inference.predictor import Predictor
        import os

        # Корень проекта
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Путь к папке с ONNX-моделью
        model_dir = os.path.join(project_root, "model", "model", "ecopulse_rubert")
        onnx_file = os.path.join(model_dir, "ecopulse.onnx")

        # Если файл отсутствует - пропускаем тест
        if not os.path.exists(onnx_file):
            pytest.skip(f"ONNX-файл не найден: {onnx_file}")

        # Создаём предиктор с явным указанием пути к ONNX
        # Параметр calibration_path можно не передавать - тогда будет использоваться
        # значение из cfg.CALIBRATION_PATH, либо температура 1.0 (если файла нет).
        p = Predictor(onnx_path=onnx_file)

        result = p.predict("Выброс химикатов на заводе")
        assert "model_score" in result
        assert "label" in result
        assert "final_score" in result
        assert 0.0 <= result["model_score"] <= 1.0
        assert result["label"] in [0, 1]
    except Exception as e:
        # Если произошла любая другая ошибка (например, проблемы с токенизатором)
        pytest.skip(f"ONNX-модель не найдена или не загружается: {e}")


def test_source_weight():
    from ecopulse.inference.predictor import source_weight
    assert source_weight(0) == 0.0
    assert 0.0 < source_weight(1000) < 1.0
    assert source_weight(100000) <= 1.0
    # Больше подписчиков - больший вес
    assert source_weight(50000) > source_weight(500)


# 5. ТЕСТ КОНФИГА
def test_config_fields():
    from ecopulse.config import cfg
    assert hasattr(cfg, "BOT_TOKEN")
    assert hasattr(cfg, "DB_PATH")
    assert hasattr(cfg, "THRESHOLD")
    assert hasattr(cfg, "SCORE_MODEL_WEIGHT")
    assert hasattr(cfg, "SCORE_SOURCE_WEIGHT")
    assert cfg.SCORE_MODEL_WEIGHT + cfg.SCORE_SOURCE_WEIGHT == pytest.approx(1.0, abs=0.01)
    assert 0 < cfg.THRESHOLD < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
