from datetime import datetime
from telegram import Bot
from  ecopulse.config import cfg
from  ecopulse.db.storage import get_todays_critical


async def send_digest():
    bot  = Bot(cfg.BOT_TOKEN)
    rows = get_todays_critical(limit=5)
    if not rows:
        await bot.send_message(cfg.ANALYST_CHAT_ID,
            "☀️ Доброе утро! Критических инцидентов за ночь не обнаружено.")
        return
    lines = [f"☀️ *Дайджест {datetime.now():%d.%m.%Y}* — топ {len(rows)}:\n"]
    for i, (pid, ch, text, pub, score) in enumerate(rows, 1):
        lines.append(f"{i}. 📢 `{ch}` | {score:.0%}\n_{text[:150].replace(chr(10),' ')}..._\n")
    await bot.send_message(cfg.ANALYST_CHAT_ID, "\n".join(lines), parse_mode="Markdown")
