import os
import json
import logging
import sqlite3
import re
from datetime import datetime

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

from ecopulse.config import cfg
from  ecopulse.db.storage import save_feedback, get_labeling_queue, submit_manual_label

logger  = logging.getLogger(__name__)
app     = Application.builder().token(cfg.BOT_TOKEN).build()
_DB     = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
              os.path.abspath(__file__)))), "ecopulse", cfg.DB_PATH)


def _score_emoji(score):
    if score >= 0.85: return "🔴"
    if score >= 0.70: return "🟠"
    return "🟡"


def _db_stats():
    if not os.path.exists(_DB):
        return {"total": 0, "critical": 0, "today": 0}
    try:
        con = sqlite3.connect(_DB)
        total    = con.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        critical = con.execute("SELECT COUNT(*) FROM posts WHERE label=1").fetchone()[0]
        today    = con.execute("SELECT COUNT(*) FROM posts WHERE date(parsed_at)=date('now')").fetchone()[0]
        con.close()
        return {"total": total, "critical": critical, "today": today}
    except Exception:
        return {"total": 0, "critical": 0, "today": 0}


def _gigachat_check(text):
    try:
        from gigachat import GigaChat
        from gigachat.models import Chat, Messages, MessagesRole
        creds = os.environ.get("GIGACHAT_CREDENTIALS", "")
        if not creds:
            return {"error": "GIGACHAT_CREDENTIALS не задан"}
        with GigaChat(credentials=creds, scope="GIGACHAT_API_PERS",
                      model="GigaChat", verify_ssl_certs=False) as giga:
            payload = Chat(messages=[
                Messages(role=MessagesRole.SYSTEM, content=(
                    "Ты эксперт по ESG-рискам. Определи: содержит ли текст критический инцидент? "
                    'Ответь строго в JSON: {"label":"critical_incident" или "not_critical",'
                    '"confidence":0.0-1.0,"reason":"одно предложение"}'
                )),
                Messages(role=MessagesRole.USER, content=f"Текст: {text}"),
            ], temperature=0.0)
            content = giga.chat(payload).choices[0].message.content
            content = re.sub(r"^```json\s*|\s*```$", "", content.strip())
            return json.loads(content)
    except ImportError:
        return {"error": "pip install gigachat"}
    except Exception as e:
        return {"error": str(e)}


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🌿 *EcoPulse* — мониторинг ESG-репутации\n\nНапиши /help",
        parse_mode="Markdown"
    )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📋 *Команды:*\n\n"
        "/status — статистика\n"
        "/digest — дайджест за сегодня\n"
        "/label — пост для разметки\n"
        "/gigachat <текст> — проверить через GigaChat\n"
        "/help — помощь",
        parse_mode="Markdown"
    )


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = _db_stats()
    model_ok = os.path.exists(os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "ecopulse", cfg.ONNX_PATH))
    await update.message.reply_text(
        f"📊 *Статус EcoPulse*\n\n"
        f"🗄 Постов в БД:    {s['total']}\n"
        f"🔴 Критических:    {s['critical']}\n"
        f"📅 Сегодня:        {s['today']}\n\n"
        f"🤖 ONNX модель: {'✅' if model_ok else '⚠️ не найдена'}\n"
        f"📱 VK API:      {'✅' if os.environ.get('VK_TOKEN') else '❌'}\n"
        f"🧠 GigaChat:    {'✅' if os.environ.get('GIGACHAT_CREDENTIALS') else '❌'}\n\n"
        f"⏰ {datetime.now().strftime('%H:%M:%S')}",
        parse_mode="Markdown"
    )


async def cmd_digest(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(_DB):
        await update.message.reply_text("БД пуста. Запусти `python main.py`.")
        return
    try:
        con  = sqlite3.connect(_DB)
        rows = con.execute("""
            SELECT channel, text, final_score FROM posts
            WHERE label=1 AND date(parsed_at)=date('now')
            ORDER BY final_score DESC LIMIT 5
        """).fetchall()
        con.close()
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")
        return
    if not rows:
        await update.message.reply_text("☀️ Критических инцидентов за сегодня нет.")
        return
    msg = f"☀️ *Дайджест {datetime.now().strftime('%d.%m.%Y')}*\n\n"
    for i, (ch, text, score) in enumerate(rows, 1):
        msg += f"{i}. 📢 `{ch}` | {score:.0%}\n_{text[:150].replace(chr(10),' ')}..._\n\n"
    await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_label(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    queue = get_labeling_queue(limit=1)
    if not queue:
        await update.message.reply_text("✅ Очередь разметки пуста!")
        return
    queue_id, text, channel, model_score, _ = queue[0]
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Критично",    callback_data=f"lbl:{queue_id}:1"),
        InlineKeyboardButton("⬜ Не критично", callback_data=f"lbl:{queue_id}:0"),
        InlineKeyboardButton("⏭ Пропустить",  callback_data=f"lbl:{queue_id}:-1"),
    ]])
    await update.message.reply_text(
        f"🏷 *Разметка*\n📢 `{channel}` | скор: {model_score:.2f}\n\n{text[:500]}",
        parse_mode="Markdown", reply_markup=kb
    )


async def cmd_gigachat(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = " ".join(ctx.args) if ctx.args else ""
    if not text:
        await update.message.reply_text("Пример: /gigachat На заводе произошёл выброс химикатов")
        return
    await update.message.reply_text("🧠 Анализирую...")
    result = _gigachat_check(text)
    if "error" in result:
        await update.message.reply_text(f"❌ {result['error']}")
        return
    emoji = "🔴" if result.get("label") == "critical_incident" else "✅"
    await update.message.reply_text(
        f"{emoji} *GigaChat*\n\n"
        f"Метка: `{result.get('label')}`\n"
        f"Уверенность: `{result.get('confidence', 0):.0%}`\n"
        f"Причина: _{result.get('reason', '—')}_",
        parse_mode="Markdown"
    )


async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data  = query.data
    if data.startswith("fb:"):
        _, post_id, is_correct = data.split(":")
        save_feedback(int(post_id), query.from_user.id, int(is_correct))
        msg = "✅ Спасибо!" if int(is_correct) else "📝 Записали ошибку."
        await query.edit_message_reply_markup(None)
        await query.message.reply_text(msg)
    elif data.startswith("lbl:"):
        _, queue_id, label = data.split(":")
        submit_manual_label(int(queue_id), int(label))
        labels = {"1": "✅ Критично", "0": "⬜ Не критично", "-1": "⏭ Пропущено"}
        await query.edit_message_reply_markup(None)
        await query.message.reply_text(f"{labels.get(label, '?')} — сохранено. /label — следующий")


async def send_alert(post_id, post, score, entities=None, trending=False):
    emoji     = _score_emoji(score)
    trend_tag = " 🔥 *Тренд!*" if trending else ""
    text      = post["text"][:300] + ("..." if len(post["text"]) > 300 else "")
    ent_lines = ""
    if entities:
        if entities.get("organizations"):
            ent_lines += f"\n🏢 {', '.join(entities['organizations'])}"
        if entities.get("locations"):
            ent_lines += f"\n📍 {', '.join(entities['locations'])}"
        if entities.get("incident_type"):
            ent_lines += f"\n⚠️ {entities['incident_type']}"
    pub = post.get("published", "")
    if hasattr(pub, "strftime"):
        pub = pub.strftime("%d.%m.%Y %H:%M")
    msg = (
        f"{emoji} *Критический инцидент*{trend_tag}\n\n"
        f"📢 `{post['channel']}`\n⏰ {pub}\n📊 {score:.0%}"
        f"{ent_lines}\n\n{text}\n\n[Источник]({post.get('url','')})"
    )
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("👍 Верно",  callback_data=f"fb:{post_id}:1"),
        InlineKeyboardButton("👎 Ошибка", callback_data=f"fb:{post_id}:0"),
    ]])
    await app.bot.send_message(
        chat_id=cfg.ANALYST_CHAT_ID, text=msg,
        parse_mode="Markdown", reply_markup=kb, disable_web_page_preview=False
    )


app.add_handler(CommandHandler("start",    cmd_start))
app.add_handler(CommandHandler("help",     cmd_help))
app.add_handler(CommandHandler("status",   cmd_status))
app.add_handler(CommandHandler("digest",   cmd_digest))
app.add_handler(CommandHandler("label",    cmd_label))
app.add_handler(CommandHandler("gigachat", cmd_gigachat))
app.add_handler(CallbackQueryHandler(handle_callback))
