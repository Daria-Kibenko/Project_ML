import asyncio
import logging
from datetime import datetime

from  ecopulse.config import cfg
from  ecopulse.parser.sources import fetch_all_sources, Post
from  ecopulse.db.storage import save_post, mark_notified
from  ecopulse.bot.bot import app as bot_app, send_alert

logger = logging.getLogger(__name__)

SOURCE_WEIGHTS = {"rss": 0.7, "vk": 0.5, "tgstat": 0.6}


def _get_predictor():
    try:
        from inference.predictor import Predictor
        return Predictor()
    except Exception as e:
        logger.warning(f"[runner] предиктор недоступен: {e} — сохраняем без классификации")
        return None


def _get_trend_tools():
    try:
        from inference.trend_detector import detect_trending_clusters, trend_boost
        return detect_trending_clusters, trend_boost
    except Exception:
        return None, None


def _get_ner():
    try:
        from inference.ner import extract_entities
        return extract_entities
    except Exception:
        return None


async def process_posts(posts, predictor):
    critical_posts = []
    for post in posts:
        sw = SOURCE_WEIGHTS.get(post.source_type, 0.5)

        if predictor:
            result = predictor.predict(post.text, source_weight=sw)
            model_score = result["model_score"]
            label       = result["label"]
            final_score = result["final_score"]
        else:
            try:
                from parser.keywords import is_priority
                label = 1 if is_priority(post.text) else 0
            except Exception:
                label = 0
            model_score = float(label)
            final_score = model_score * cfg.SCORE_MODEL_WEIGHT + sw * cfg.SCORE_SOURCE_WEIGHT

        post_id = save_post(
            channel=post.source,
            message_id=post.post_id,
            text=post.text,
            published=post.published,
            model_score=model_score,
            source_weight=sw,
            label=label,
        )

        if label == 1 and post_id:
            critical_posts.append({
                "post_id":     post_id,
                "text":        post.text,
                "channel":     post.source,
                "published":   post.published,
                "url":         post.url,
                "final_score": final_score,
            })

    if not critical_posts:
        return

    detect_clusters, trend_boost_fn = _get_trend_tools()
    if detect_clusters:
        critical_posts = detect_clusters(critical_posts)

    extract_entities = _get_ner()

    for post in critical_posts:
        if trend_boost_fn:
            boost = trend_boost_fn(post.get("cluster_size", 1))
            post["final_score"] = min(1.0, post["final_score"] + boost)

        entities = extract_entities(post["text"]) if extract_entities else {}
        trending = post.get("cluster_size", 1) > 1

        await send_alert(post["post_id"], post, post["final_score"], entities, trending)
        mark_notified(post["post_id"])

    logger.info(f"[runner] отправлено уведомлений: {len(critical_posts)}")


async def run_once():
    predictor = _get_predictor()
    logger.info(f"[runner] старт {datetime.now():%H:%M:%S}")
    posts = fetch_all_sources()
    logger.info(f"[runner] получено постов: {len(posts)}")
    await process_posts(posts, predictor)


async def run_scheduler():
    while True:
        try:
            await run_once()
        except Exception as e:
            logger.error(f"[runner] ошибка: {e}")
        await asyncio.sleep(cfg.PARSE_INTERVAL_MINUTES * 60)
