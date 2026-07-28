import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import asyncio
import logging
import threading

os.makedirs(os.path.join(_PROJECT_ROOT, "logs"), exist_ok=True)
os.makedirs(os.path.join(_PROJECT_ROOT, "db"),   exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(_PROJECT_ROOT, "logs", "ecopulse.log"),
            encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger(__name__)

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from ecopulse.config import cfg
from ecopulse.db.storage import init_db
from ecopulse.parser.runner import run_once
from ecopulse.bot.digest import send_digest
from ecopulse.bot.retrain_trigger import check_and_retrain
from ecopulse.monitoring.drift import check_drift


def run_bot_in_thread():
    from ecopulse.bot.bot import app as bot_app

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _run():
        await bot_app.initialize()
        await bot_app.start()
        await bot_app.updater.start_polling(drop_pending_updates=True)
        logger.info("✅ Telegram-бот запущен")
        # Ждём пока не придёт сигнал остановки
        stop_event = asyncio.Event()
        try:
            await stop_event.wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await bot_app.updater.stop()
            await bot_app.stop()
            await bot_app.shutdown()

    try:
        loop.run_until_complete(_run())
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        loop.close()


async def parser_loop():
    from ecopulse.parser.runner import run_once
    while True:
        try:
            await run_once()
        except Exception as e:
            logger.error(f"[parser] ошибка: {e}")
        await asyncio.sleep(cfg.PARSE_INTERVAL_MINUTES * 60)


async def main():
    # Проверка конфига
    errors = []
    if not cfg.BOT_TOKEN or cfg.BOT_TOKEN in ("your_bot_token", ""):
        errors.append("BOT_TOKEN не заполнен в ecopulse/config.py")
    if cfg.ANALYST_CHAT_ID == 0:
        errors.append("ANALYST_CHAT_ID не заполнен — напиши @userinfobot")
    if errors:
        for e in errors:
            logger.error(f"❌ {e}")
        return

    init_db()

    ecopulse_dir = os.path.join(_PROJECT_ROOT, "ecopulse")
    model_ready  = os.path.exists(os.path.join(ecopulse_dir, cfg.ONNX_PATH))
    vk_ok        = bool(getattr(cfg, "VK_TOKEN", "") or os.environ.get("VK_TOKEN"))
    giga_ok      = bool(os.environ.get("GIGACHAT_CREDENTIALS"))

    logger.info(
        f"\n{'='*50}\n"
        f"  🌿 EcoPulse запускается\n"
        f"  ├─ Парсинг каждые {cfg.PARSE_INTERVAL_MINUTES} мин\n"
        f"  ├─ Дайджест в {cfg.DIGEST_HOUR}:00\n"
        f"  ├─ VK API:      {'✅' if vk_ok       else '❌ нет токена'}\n"
        f"  ├─ GigaChat:    {'✅' if giga_ok      else '⚠️  нет ключа'}\n"
        f"  ├─ ONNX модель: {'✅' if model_ready  else '⚠️  не найдена'}\n"
        f"  └─ Бот:         @EcoPulse26Bot\n"
        f"{'='*50}"
    )

    # Бот — в отдельном потоке
    bot_thread = threading.Thread(target=run_bot_in_thread, daemon=True)
    bot_thread.start()

    # Планировщик
    scheduler = AsyncIOScheduler()
    scheduler.add_job(send_digest,       "cron", hour=cfg.DIGEST_HOUR, minute=0)
    scheduler.add_job(check_and_retrain, "cron", hour=2, minute=0)
    scheduler.add_job(check_drift,       "cron", day_of_week="mon", hour=3, minute=0)
    scheduler.start()
    logger.info("✅ Планировщик запущен")

    # Парсер - в главном event loop
    await parser_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("EcoPulse остановлен.")