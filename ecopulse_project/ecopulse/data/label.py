import json
import time
import os
import re
import pandas as pd
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

# Настройки
GIGACHAT_CREDENTIALS = os.environ.get(
    "GIGACHAT_CREDENTIALS",
    "MDE5YjcwYzAtODU4Ny03ZWQ3LWIzNzAtYWY4NTRiYzUxNzc1OmY3ODUxNDgxLWQyZjYtNDZkMy05OGE2LTFjODFiM2JjNWRjYg=="
)

# scope: GIGACHAT_API_PERS - для личного аккаунта (бесплатный тариф)
#        GIGACHAT_API_CORP - для корпоративного
GIGACHAT_SCOPE = os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")

# Модель: GigaChat (бесплатная), GigaChat-Pro (платная, лучше качество)
GIGACHAT_MODEL = os.environ.get("GIGACHAT_MODEL", "GigaChat")

# Промпт
SYSTEM_PROMPT = """Ты - эксперт по ESG-репутационным рискам для промышленных компаний.

Критический инцидент - событие, способное нанести ущерб:
- здоровью или жизни людей (аварии, отравления, травмы)
- окружающей среде (разливы, выбросы, загрязнения)
- репутации компании в значимом масштабе (обман клиентов, утечка данных,
  нарушение прав работников)

Ответь строго в формате JSON - без текста вне JSON, без markdown:
{"label": "critical_incident" или "not_critical", "confidence": 0.0-1.0, "reason": "одно предложение"}
"""


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    return json.loads(text.strip())


def label_one(text: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            with GigaChat(
                credentials=GIGACHAT_CREDENTIALS,
                scope=GIGACHAT_SCOPE,
                model=GIGACHAT_MODEL,
                verify_ssl_certs=False,   # отключаем SSL, если нет сертификата Минцифры
            ) as giga:
                payload = Chat(
                    messages=[
                        Messages(role=MessagesRole.SYSTEM, content=SYSTEM_PROMPT),
                        Messages(role=MessagesRole.USER,   content=f"Текст: {text}"),
                    ],
                    temperature=0.0,
                )
                response = giga.chat(payload)
                content = response.choices[0].message.content
                return _parse_json(content)

        except json.JSONDecodeError as e:
            print(f"[label] попытка {attempt+1}: не удалось распарсить JSON: {e}")
            print(f"         ответ модели: {content[:200]}")
        except Exception as e:
            print(f"[label] попытка {attempt+1} не удалась: {e}")
            time.sleep(2 ** attempt)

    return {"label": "not_critical", "confidence": 0.0, "reason": "error"}


def label_uncertain_batch(
    df: pd.DataFrame,
    min_conf: float = 0.80,
    save_progress: str = "data/llm_progress.csv",
) -> pd.DataFrame:
    results = []

    # Если есть прогресс - подгружаем и пропускаем уже обработанные
    already_done = set()
    if os.path.exists(save_progress):
        prev = pd.read_csv(save_progress)
        already_done = set(prev.index)
        results = prev.to_dict("records")
        print(f"[label] найден прогресс: {len(results)} уже размечено, продолжаем...")

    for i, (idx, row) in enumerate(df.iterrows()):
        if idx in already_done:
            continue

        res = label_one(row["text"])
        results.append({
            **row.to_dict(),
            "llm_label":      1 if res["label"] == "critical_incident" else 0,
            "llm_confidence": res["confidence"],
            "llm_reason":     res["reason"],
        })

        if (i + 1) % 10 == 0:
            print(f"[label] обработано {i + 1}/{len(df)}")
            pd.DataFrame(results).to_csv(save_progress, index=False)

    pd.DataFrame(results).to_csv(save_progress, index=False)

    out = pd.DataFrame(results)
    confident = out[out["llm_confidence"] >= min_conf].copy()
    confident["label"] = confident["llm_label"]

    print(f"\n[llm-label] итого: {len(out)} обработано")
    print(f"[llm-label] принято (confidence >= {min_conf}): {len(confident)}")
    print(f"[llm-label] critical: {confident['label'].sum()}, "
          f"not_critical: {(confident['label'] == 0).sum()}")

    return confident.drop(columns=["llm_label", "llm_confidence", "llm_reason"])


def test_connection():
    print("=== Тест подключения к GigaChat ===")
    print(f"Ключ: {GIGACHAT_CREDENTIALS[:8]}...{GIGACHAT_CREDENTIALS[-4:]}")
    print(f"Scope: {GIGACHAT_SCOPE}")
    print(f"Модель: {GIGACHAT_MODEL}\n")

    test_text = "На заводе НЛМК в Липецке произошёл разлив химикатов, жители жалуются на запах"
    print(f"Тестовый текст: {test_text}\n")

    result = label_one(test_text)
    print(f"Результат: {json.dumps(result, ensure_ascii=False, indent=2)}")

    if result["label"] in ("critical_incident", "not_critical") and result["confidence"] > 0:
        print("\n✅ GigaChat работает корректно!")
    else:
        print("\n❌ Что-то пошло не так, проверь ключ и scope.")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv or len(sys.argv) == 1:
        test_connection()
    else:
        # Полный прогон на датасете
        df = pd.read_csv("data/dataset_combined.csv")
        # Для теста берём 100 строк, для продакшна убери .sample()
        sample = df.sample(100, random_state=42)
        labeled = label_uncertain_batch(sample)
        labeled.to_csv("data/llm_verified_sample.csv", index=False)
        print(f"\nГотово: {len(labeled)} строк сохранено → data/llm_verified_sample.csv")
