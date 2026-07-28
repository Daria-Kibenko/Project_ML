import os
import json
import re
import logging

logger = logging.getLogger(__name__)


FORECAST_PROMPT = """Ты — эксперт по репутационным рискам и социологии.

Тебе дан текст публикации о компании или событии. Твоя задача:
1. Оценить вероятную реакцию людей на эту публикацию
2. Придумать 3 типичных комментария которые могут написать пользователи
3. Дать прогноз по уровню репутационного риска

Ответь строго в JSON без markdown:
{
  "sentiment": "позитивный" или "негативный" или "нейтральный",
  "reaction_intensity": "низкая" или "средняя" или "высокая",
  "risk_level": "низкий" или "средний" или "высокий" или "критический",
  "predicted_comments": ["комментарий 1", "комментарий 2", "комментарий 3"],
  "risk_explanation": "краткое объяснение почему такой уровень риска",
  "recommended_action": "что рекомендуется сделать PR-команде"
}
"""


def generate_forecast(text: str) -> dict:
    creds = os.environ.get("GIGACHAT_CREDENTIALS", "")
    if not creds:
        return {"error": "GIGACHAT_CREDENTIALS не задан"}

    try:
        from gigachat import GigaChat
        from gigachat.models import Chat, Messages, MessagesRole

        with GigaChat(credentials=creds, scope="GIGACHAT_API_PERS",
                      model="GigaChat", verify_ssl_certs=False) as giga:
            payload = Chat(
                messages=[
                    Messages(role=MessagesRole.SYSTEM, content=FORECAST_PROMPT),
                    Messages(role=MessagesRole.USER,
                             content=f"Публикация:\n{text[:2000]}"),
                ],
                temperature=0.3,
                max_tokens=800,
            )
            content = giga.chat(payload).choices[0].message.content
            content = re.sub(r"^```json\s*|\s*```$", "", content.strip())
            return json.loads(content)

    except ImportError:
        return {"error": "pip install gigachat"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "raw": content[:300]}
    except Exception as e:
        return {"error": str(e)}


def format_forecast_message(post_text: str, forecast: dict) -> str:
    """Форматирует прогноз для отправки в Telegram-бот."""
    if "error" in forecast:
        return f"⚠️ GigaChat недоступен: {forecast['error']}"

    risk_emoji = {
        "низкий": "🟢", "средний": "🟡",
        "высокий": "🟠", "критический": "🔴"
    }.get(forecast.get("risk_level", ""), "⚪")

    sentiment_emoji = {
        "позитивный": "😊", "негативный": "😠", "нейтральный": "😐"
    }.get(forecast.get("sentiment", ""), "")

    comments = "\n".join(
        f"  • _{c}_"
        for c in forecast.get("predicted_comments", [])[:3]
    )

    return (
        f"🧠 *Прогноз GigaChat*\n\n"
        f"{sentiment_emoji} Тональность: `{forecast.get('sentiment', '—')}`\n"
        f"{risk_emoji} Репутационный риск: `{forecast.get('risk_level', '—')}`\n"
        f"📊 Интенсивность реакции: `{forecast.get('reaction_intensity', '—')}`\n\n"
        f"💬 *Вероятные комментарии:*\n{comments}\n\n"
        f"⚠️ *Пояснение:* {forecast.get('risk_explanation', '—')}\n\n"
        f"✅ *Рекомендация:* {forecast.get('recommended_action', '—')}"
    )
