import sys
import os
import time
import sqlite3
import logging

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ecopulse.config import cfg

_DB = os.path.join(_ROOT, "ecopulse", cfg.DB_PATH)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WORKER %(process)d] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

WORKER_INTERVAL = 30   # секунд между циклами


def get_unclassified(limit: int = 50) -> list:
    if not os.path.exists(_DB):
        return []
    try:
        con = sqlite3.connect(_DB)
        rows = con.execute(f"""
            SELECT id, text, source_weight
            FROM posts
            WHERE model_score IS NULL
            LIMIT {limit}
        """).fetchall()
        con.close()
        return rows
    except Exception as e:
        logger.error(f"Ошибка чтения БД: {e}")
        return []


def update_post(post_id: int, model_score: float,
                final_score: float, label: int):
    try:
        con = sqlite3.connect(_DB)
        con.execute("""
            UPDATE posts
            SET model_score=?, final_score=?, label=?
            WHERE id=?
        """, (model_score, final_score, label, post_id))
        con.commit()
        con.close()
    except Exception as e:
        logger.error(f"Ошибка обновления поста {post_id}: {e}")


def run_worker():
    logger.info(f"Воркер запущен (PID={os.getpid()})")
    logger.info(f"БД: {_DB}")

    predictor = None
    try:
        from inference.predictor import Predictor
        predictor = Predictor()
        logger.info("✅ ONNX-предиктор загружен")
    except Exception as e:
        logger.warning(f"⚠️  Предиктор недоступен ({e}) — используем ключевые слова")

    while True:
        posts = get_unclassified(limit=50)

        if not posts:
            logger.debug("Нет необработанных постов — ждём...")
            time.sleep(WORKER_INTERVAL)
            continue

        logger.info(f"Классифицируем {len(posts)} постов...")
        classified = 0

        for post_id, text, source_weight in posts:
            try:
                sw = float(source_weight) if source_weight else 0.5

                if predictor:
                    result      = predictor.predict(text, source_weight=sw)
                    model_score = result["model_score"]
                    final_score = result["final_score"]
                    label       = result["label"]
                else:
                    # fallback: ключевые слова
                    try:
                        from parser.keywords import is_priority
                        label = 1 if is_priority(text) else 0
                    except Exception:
                        label = 0
                    model_score = 0.75 if label else 0.2
                    final_score = (model_score * cfg.SCORE_MODEL_WEIGHT
                                   + sw * cfg.SCORE_SOURCE_WEIGHT)

                update_post(post_id, model_score, final_score, label)
                classified += 1

            except Exception as e:
                logger.error(f"Ошибка при классификации поста {post_id}: {e}")

        logger.info(f"Готово: {classified}/{len(posts)} постов классифицировано")
        time.sleep(WORKER_INTERVAL)


if __name__ == "__main__":
    run_worker()
