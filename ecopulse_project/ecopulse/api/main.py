import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
import re
import sqlite3
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ecopulse.config import cfg
from ecopulse.db.storage import (
    save_feedback, get_labeling_queue,
    submit_manual_label, get_todays_critical
)

_DB = os.path.join(_ROOT, "ecopulse", cfg.DB_PATH)

app = FastAPI(
    title="EcoPulse API",
    description="Мониторинг ESG-репутации — REST интерфейс",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic схемы
class ClassifyRequest(BaseModel):
    text: str
    source_weight: float = 0.5

class ClassifyResponse(BaseModel):
    text: str
    model_score: float
    final_score: float
    label: int
    label_text: str
    incident_category: str

class ForecastRequest(BaseModel):
    text: str

class ForecastResponse(BaseModel):
    sentiment: str
    reaction_intensity: str
    risk_level: str
    predicted_comments: List[str]
    risk_explanation: str
    recommended_action: str

class FeedbackRequest(BaseModel):
    post_id: int
    user_id: int
    is_correct: int  # 1 = верно, 0 = ошибка

class LabelRequest(BaseModel):
    label: int  # 1=critical, 0=not_critical, -1=skip


# Утилиты
def _get_predictor():
    try:
        from inference.predictor import Predictor
        return Predictor()
    except Exception:
        return None


def _db_query(sql: str, params: tuple = ()) -> list:
    if not os.path.exists(_DB):
        return []
    con = sqlite3.connect(_DB)
    con.row_factory = sqlite3.Row
    rows = con.execute(sql, params).fetchall()
    con.close()
    return [dict(r) for r in rows]


# Эндпоинты
@app.get("/health", tags=["Система"])
def health():
    model_ok = os.path.exists(os.path.join(_ROOT, "ecopulse", cfg.ONNX_PATH))
    db_ok    = os.path.exists(_DB)
    return {
        "status":    "ok",
        "timestamp": datetime.now().isoformat(),
        "model":     "ready" if model_ok else "not_found",
        "database":  "ready" if db_ok else "not_found",
        "version":   "1.0.0",
    }


@app.get("/stats", tags=["Аналитика"])
def stats():
    if not os.path.exists(_DB):
        return {"total": 0, "critical": 0, "today": 0}
    rows = _db_query("""
        SELECT
            COUNT(*) as total,
            SUM(label) as critical,
            SUM(CASE WHEN date(parsed_at)=date('now') THEN 1 ELSE 0 END) as today,
            AVG(final_score) as avg_score,
            COUNT(DISTINCT channel) as channels
        FROM posts
    """)
    return rows[0] if rows else {}


@app.get("/posts", tags=["Посты"])
def get_posts(
    days:     int   = Query(7,  ge=1, le=90, description="Глубина в днях"),
    label:    Optional[int] = Query(None, description="0=обычные, 1=критические"),
    source:   Optional[str] = Query(None, description="Фильтр по каналу"),
    limit:    int   = Query(50, ge=1, le=500),
):
    sql    = "SELECT id,channel,text,published,final_score,label,parsed_at FROM posts WHERE parsed_at >= date('now', ?)"
    params = [f"-{days} days"]
    if label is not None:
        sql += " AND label=?"; params.append(label)
    if source:
        sql += " AND channel LIKE ?"; params.append(f"%{source}%")
    sql += " ORDER BY parsed_at DESC LIMIT ?"
    params.append(limit)
    return _db_query(sql, tuple(params))


@app.get("/posts/critical", tags=["Посты"])
def get_critical(limit: int = Query(20, ge=1, le=200)):
    rows = get_todays_critical(limit=limit)
    return [{"id":r[0],"channel":r[1],"text":r[2],
             "published":str(r[3]),"final_score":r[4]} for r in rows]


@app.post("/posts/classify", response_model=ClassifyResponse, tags=["Модель"])
def classify_text(req: ClassifyRequest):
    predictor = _get_predictor()
    if predictor is None:
        # fallback на ключевые слова
        try:
            from parser.keywords import is_priority, get_incident_category
            lbl  = 1 if is_priority(req.text) else 0
            cat  = get_incident_category(req.text)
            sc   = 0.75 if lbl else 0.25
        except Exception:
            lbl, sc, cat = 0, 0.5, "не определён"
        return ClassifyResponse(
            text=req.text, model_score=sc, final_score=sc,
            label=lbl, label_text="critical" if lbl else "not_critical",
            incident_category=cat,
        )

    result = predictor.predict(req.text, source_weight=req.source_weight)
    try:
        from parser.keywords import get_incident_category
        cat = get_incident_category(req.text)
    except Exception:
        cat = "не определён"

    return ClassifyResponse(
        text=req.text,
        model_score=result["model_score"],
        final_score=result["final_score"],
        label=result["label"],
        label_text="critical" if result["label"] else "not_critical",
        incident_category=cat,
    )


@app.post("/posts/forecast", response_model=ForecastResponse, tags=["GigaChat"])
def forecast_reaction(req: ForecastRequest):
    creds = (os.environ.get("GIGACHAT_CREDENTIALS") or
             getattr(cfg, "GIGACHAT_CREDENTIALS", ""))
    if not creds:
        # демо-ответ
        return ForecastResponse(
            sentiment="негативный",
            reaction_intensity="высокая",
            risk_level="критический",
            predicted_comments=[
                "Опять нарушения — когда это прекратится?",
                "Требуем проверки и наказания виновных!",
                "Будем следить за результатами расследования.",
            ],
            risk_explanation="Текст содержит признаки ESG-инцидента с высоким репутационным риском.",
            recommended_action="Немедленно подготовить официальное заявление.",
        )

    try:
        from gigachat import GigaChat
        from gigachat.models import Chat, Messages, MessagesRole

        PROMPT = (
            "Ты эксперт по ESG-рискам. Оцени публикацию. "
            "Ответь строго в JSON без markdown: "
            '{"sentiment":"...","reaction_intensity":"...","risk_level":"...",'
            '"predicted_comments":["...","...","..."],'
            '"risk_explanation":"...","recommended_action":"..."}'
        )
        with GigaChat(credentials=creds, scope="GIGACHAT_API_PERS",
                       model="GigaChat", verify_ssl_certs=False) as g:
            resp = g.chat(Chat(messages=[
                Messages(role=MessagesRole.SYSTEM, content=PROMPT),
                Messages(role=MessagesRole.USER, content=f"Текст: {req.text[:2000]}"),
            ], temperature=0.3, max_tokens=600))
        content = re.sub(r"^```json\s*|\s*```$", "",
                          resp.choices[0].message.content.strip())
        data = json.loads(content)
        return ForecastResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback", tags=["Фидбек"])
def add_feedback(req: FeedbackRequest):
    save_feedback(req.post_id, req.user_id, req.is_correct)
    return {"status": "ok", "post_id": req.post_id}


@app.get("/labeling/queue", tags=["Active Learning"])
def labeling_queue(limit: int = Query(20, ge=1, le=100)):
    rows = get_labeling_queue(limit=limit)
    return [{"queue_id":r[0],"text":r[1],"channel":r[2],
             "model_score":r[3],"uncertainty":r[4]} for r in rows]


@app.post("/labeling/{queue_id}", tags=["Active Learning"])
def label_post(queue_id: int, req: LabelRequest):
    submit_manual_label(queue_id, req.label)
    return {"status": "ok", "queue_id": queue_id, "label": req.label}
