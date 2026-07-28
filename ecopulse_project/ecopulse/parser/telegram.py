import asyncio
from datetime import datetime, timezone
from telethon import TelegramClient
from telethon.tl.types import Message

from ecopulse.config import cfg
from inference.predictor import Predictor, source_weight
from inference.trend_detector import detect_trending_clusters, trend_boost
from inference.ner import extract_entities
from db.storage import save_post, mark_notified
from bot.bot import send_alert

client = TelegramClient(cfg.SESSION_NAME, cfg.API_ID, cfg.API_HASH)
predictor = Predictor()


async def parse_channel(channel: str, since_minutes: int = 20) -> list:
    posts = []
    async for msg in client.iter_messages(channel, limit=100):
        if not isinstance(msg, Message) or not msg.text:
            continue
        age = (datetime.now(timezone.utc) - msg.date).total_seconds() / 60
        if age > since_minutes:
            break
        posts.append({
            "channel": channel,
            "message_id": msg.id,
            "text": msg.text,
            "published": msg.date,
            "url": f"https://t.me/{channel}/{msg.id}",
        })
    return posts


async def run_once():
    all_critical_posts = []

    async with client:
        for channel, subscribers in cfg.CHANNELS.items():
            try:
                posts = await parse_channel(channel)
            except Exception as e:
                print(f"[parser] ошибка {channel}: {e}")
                continue

            sw = source_weight(subscribers)

            for post in posts:
                result = predictor.predict(post["text"], source_weight=sw)
                post_id = save_post(
                    post["channel"], post["message_id"], post["text"],
                    post["published"], result["model_score"],
                    sw, result["label"],
                )
                if result["label"] == 1 and post_id:
                    post.update(result)
                    post["post_id"] = post_id
                    all_critical_posts.append(post)

    # трендовый анализ: усиливаем приоритет, если несколько каналов пишут об одном
    if all_critical_posts:
        all_critical_posts = detect_trending_clusters(all_critical_posts)

        for post in all_critical_posts:
            boost = trend_boost(post["cluster_size"])
            post["final_score"] = min(1.0, post["final_score"] + boost)

            entities = extract_entities(post["text"])
            post["entities"] = entities

            await send_alert(post["post_id"], post, post["final_score"], entities,
                              trending=post["cluster_size"] > 1)
            mark_notified(post["post_id"])


async def run_scheduler():
    while True:
        print(f"[scheduler] старт парсинга {datetime.now():%H:%M:%S}")
        try:
            await run_once()
        except Exception as e:
            print(f"[scheduler] ошибка цикла: {e}")
        await asyncio.sleep(cfg.PARSE_INTERVAL_MINUTES * 60)
