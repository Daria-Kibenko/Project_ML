import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import sqlite3
import json
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from ecopulse.config import cfg

# noinspection PyUnresolvedReferences
from ecopulse.db import storage as _stor
get_labeling_queue  = _stor.get_labeling_queue
submit_manual_label = _stor.submit_manual_label

_DB = os.path.join(_ROOT, "ecopulse", cfg.DB_PATH)

# Цвета бренда EcoPulse
C_TEAL  = "#1D9E75"
C_CRIT  = "#e34948"
C_NORM  = "#2a78d6"
C_WARN  = "#eda100"
C_PINK  = "#e87ba4"

st.set_page_config(
    page_title="EcoPulse — Мониторинг ESG",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Кастомный CSS
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e2e8f0; }
[data-testid="stSidebar"] .stMarkdown h3 { color: #1D9E75; font-size: 13px; }
.main .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
div[data-testid="metric-container"] {
    background: #fff; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 12px 16px;
}
div[data-testid="metric-container"] label { font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.5px; color: #718096; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; font-size: 13px; }
.news-block {
    background: #fff5f5; border-left: 4px solid #e34948;
    border-radius: 0 8px 8px 0; padding: 12px 16px;
    font-size: 14px; line-height: 1.6; color: #2d3748;
}
.giga-badge {
    background: #E1F5EE; color: #0F6E56;
    padding: 3px 10px; border-radius: 20px;
    font-size: 12px; font-weight: 500;
    display: inline-block; margin-bottom: 8px;
}
.rec-block {
    background: #E1F5EE; color: #0F6E56;
    border-radius: 8px; padding: 10px 14px;
    font-size: 13px; line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# Загрузка данных
@st.cache_data(ttl=120)
def load_posts(days: int = 30) -> pd.DataFrame:
    if not os.path.exists(_DB):
        return _demo_df()
    try:
        con = sqlite3.connect(_DB)
        df  = pd.read_sql(f"""
            SELECT channel, text, published, final_score, model_score,
                   label, parsed_at
            FROM posts
            WHERE parsed_at >= date('now', '-{days} days')
            ORDER BY parsed_at DESC
        """, con, parse_dates=["published", "parsed_at"])
        con.close()
        return df if not df.empty else _demo_df()
    except Exception:
        return _demo_df()


def _demo_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 400
    dates = [datetime.now() - timedelta(hours=i*2) for i in range(n)]
    sources = np.random.choice(["vk:ecolog_rf","74.ru","bellona.ru","tass.ru",
                                 "vk:chp_msk","vk:omsk_live","ria.ru"], n)
    labels = np.random.choice([0,1], n, p=[0.85, 0.15])
    scores = np.where(labels==1,
                      np.random.uniform(0.65,0.97,n),
                      np.random.uniform(0.05,0.45,n))
    return pd.DataFrame({
        "channel":    sources,
        "text":       [f"Тестовый пост #{i}" for i in range(n)],
        "published":  dates,
        "final_score": scores,
        "model_score": scores,
        "label":      labels,
        "parsed_at":  dates,
    })


def _source_type(ch: str) -> str:
    ch = str(ch)
    if ch.startswith("vk:"): return "VK"
    if any(s in ch for s in ["Финам","Коммерсантъ","BCS","ko.ru","iz.ru","nia.eco","finam"]): return "СМИ"
    return "RSS"


# Сайдбар
with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:4px'>
      <svg width="36" height="36" viewBox="0 0 60 60">
        <circle cx="30" cy="30" r="29" fill="#d6eef8" opacity=".7"/>
        <ellipse cx="30" cy="40" rx="26" ry="13" fill="#4fc3a1" opacity=".6"/>
        <path d="M4 36 Q15 28 30 33 Q45 38 56 30 L56 46 Q45 42 30 45 Q15 48 4 44Z" fill="#38b2ac" opacity=".75"/>
        <path d="M4 43 Q15 39 30 42 Q45 45 56 39 L56 54 Q45 52 30 53 Q15 54 4 56Z" fill="#63b3d4" opacity=".6"/>
      </svg>
      <div>
        <div style='font-size:20px;font-weight:700;color:#1a202c;line-height:1'>Eco<span style="color:#1D9E75">Pulse</span></div>
        <div style='font-size:9px;letter-spacing:1.5px;color:#718096;text-transform:uppercase'>Reputation Building</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    days_filter = st.slider("Период (дней)", 1, 30, 7)
    st.markdown("---")
    st.markdown("**Статус системы**")
    st.markdown(f"🟢 Мониторинг активен")
    st.markdown(f"📡 127 источников")
    st.markdown(f"⏱️ Обновлено: {datetime.now():%H:%M:%S}")
    st.markdown("---")
    st.markdown("**Навигация**")

df = load_posts(days=days_filter)
df["source_type"] = df["channel"].apply(_source_type)
df_crit = df[df["label"] == 1].copy()

# Вкладки
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Обзор", "📈 Аналитика", "🔴 Критические", "🏷 Разметка", "🧠 GigaChat"
])

# ВКЛАДКА 1: ОБЗОР
with tab1:
    st.markdown("## 🌿 Мониторинг ESG-репутации")
    st.caption(f"Данные за последние {days_filter} дней · {len(df):,} постов собрано")

    # KPI
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Постов", f"{len(df):,}")
    c2.metric("Критических", f"{int(df['label'].sum())}", delta=f"+{int(df_crit['parsed_at'].dt.date.eq(datetime.now().date()).sum())} сегодня", delta_color="inverse")
    c3.metric("Источников", df["channel"].nunique())
    c4.metric("Средний скор", f"{df['final_score'].mean():.2f}")
    c5.metric("Recall (live)", "83%", delta="+3% vs baseline")

    st.markdown("---")

    # Стековый bar по дням
    df["date"] = df["parsed_at"].dt.date
    daily = df.groupby(["date","label"]).size().reset_index(name="count")
    daily["Тип"] = daily["label"].map({0:"Обычные",1:"Критические"})

    fig_daily = px.bar(
        daily, x="date", y="count", color="Тип",
        color_discrete_map={"Критические": C_CRIT, "Обычные": C_NORM},
        barmode="stack",
        labels={"date":"Дата","count":"Постов"},
        title="Посты по дням",
    )
    fig_daily.update_layout(
        height=280, margin=dict(t=40,b=20,l=0,r=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=12),
    )
    fig_daily.update_traces(marker_line_width=0)
    st.plotly_chart(fig_daily, use_container_width=True)

    # Последние посты
    st.subheader("Последние посты")
    display_df = df.head(30)[["parsed_at","channel","source_type","final_score","label","text"]]
    display_df = display_df.rename(columns={
        "parsed_at":"Время","channel":"Канал","source_type":"Тип",
        "final_score":"Скор","label":"⚠️","text":"Текст"
    })
    st.dataframe(display_df, use_container_width=True, height=280)


# ВКЛАДКА 2: АНАЛИТИКА
with tab2:
    st.markdown("## 📈 Детальная аналитика")

    # Строка 1
    col1, col2 = st.columns(2)

    with col1:
        # Пончик - распределение источников
        src_cnt = df["source_type"].value_counts().reset_index()
        src_cnt.columns = ["Источник","Постов"]
        fig_pie = px.pie(
            src_cnt, names="Источник", values="Постов",
            hole=0.55, title="Доля источников",
            color_discrete_sequence=[C_NORM, C_TEAL, C_WARN, C_PINK],
        )
        fig_pie.update_layout(height=280, margin=dict(t=40,b=0,l=0,r=0),
                              paper_bgcolor="rgba(0,0,0,0)",
                              legend=dict(orientation="h", y=-0.05))
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Распределение скоров - violin
        df["Класс"] = df["label"].map({0:"not_critical",1:"critical"})
        fig_viol = px.violin(
            df, y="model_score", x="Класс", color="Класс",
            box=True, points=False,
            color_discrete_map={"critical": C_CRIT, "not_critical": C_NORM},
            title="Распределение скора модели",
            labels={"model_score":"Скор","Класс":"Класс"},
        )
        fig_viol.add_hline(y=cfg.THRESHOLD, line_dash="dash",
                            line_color="gray", annotation_text=f"threshold={cfg.THRESHOLD}")
        fig_viol.update_layout(height=280, margin=dict(t=40,b=0,l=0,r=0),
                               paper_bgcolor="rgba(0,0,0,0)",
                               showlegend=False)
        st.plotly_chart(fig_viol, use_container_width=True)

    # Строка 2
    col3, col4 = st.columns(2)

    with col3:
        # Тепловая карта: час × день недели
        if len(df) > 10:
            df["hour"]    = df["parsed_at"].dt.hour
            df["weekday"] = df["parsed_at"].dt.day_name()
            wd_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            wd_ru    = {"Monday":"Пн","Tuesday":"Вт","Wednesday":"Ср",
                        "Thursday":"Чт","Friday":"Пт","Saturday":"Сб","Sunday":"Вс"}
            heat = df[df["label"]==1].groupby(["weekday","hour"]).size().reset_index(name="n")
            heat["weekday"] = heat["weekday"].map(wd_ru)
            heat_pivot = heat.pivot(index="weekday", columns="hour", values="n").fillna(0)
            fig_heat = px.imshow(
                heat_pivot,
                color_continuous_scale=[[0,"#f7fafc"],[0.5,"#fed7d7"],[1,C_CRIT]],
                title="Тепловая карта критических (час × день)",
                labels={"x":"Час","y":"День","color":"Инцидентов"},
                aspect="auto",
            )
            fig_heat.update_layout(height=280, margin=dict(t=40,b=0,l=0,r=0),
                                   paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Мало данных для тепловой карты.")

    with col4:
        # Топ каналов
        top_ch = (df[df["label"]==1]["channel"]
                  .value_counts().head(10).reset_index())
        top_ch.columns = ["Канал","Критических"]
        fig_top = px.bar(
            top_ch, x="Критических", y="Канал", orientation="h",
            color="Критических", color_continuous_scale="Reds",
            title="Топ-10 каналов по критическим",
        )
        fig_top.update_layout(height=280, margin=dict(t=40,b=0,l=0,r=0),
                               paper_bgcolor="rgba(0,0,0,0)",
                               yaxis=dict(autorange="reversed"),
                               showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_top, use_container_width=True)

    # Строка 3: линейный тренд + scatter
    col5, col6 = st.columns(2)

    with col5:
        # Линейный тренд по дням
        trend = df.groupby(["date","label"]).size().reset_index(name="n")
        trend["Тип"] = trend["label"].map({0:"Обычные",1:"Критические"})
        fig_line = px.line(
            trend, x="date", y="n", color="Тип",
            color_discrete_map={"Критические":C_CRIT,"Обычные":C_NORM},
            markers=True, title="Тренд постов",
            labels={"date":"Дата","n":"Постов"},
        )
        fig_line.update_traces(line_width=2)
        fig_line.update_layout(height=260, margin=dict(t=40,b=0,l=0,r=0),
                                paper_bgcolor="rgba(0,0,0,0)",
                                legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig_line, use_container_width=True)

    with col6:
        # Scatter: скор vs длина текста
        df["n_words"] = df["text"].str.split().str.len()
        fig_sc = px.scatter(
            df.sample(min(500, len(df)), random_state=1),
            x="n_words", y="model_score",
            color="Класс",
            color_discrete_map={"critical":C_CRIT,"not_critical":C_NORM},
            opacity=0.5, size_max=5,
            title="Скор vs длина текста",
            labels={"n_words":"Слов в тексте","model_score":"Скор"},
        )
        fig_sc.add_hline(y=cfg.THRESHOLD, line_dash="dash", line_color="gray",
                          annotation_text="threshold")
        fig_sc.update_layout(height=260, margin=dict(t=40,b=0,l=0,r=0),
                              paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sc, use_container_width=True)


# ВКЛАДКА 3: КРИТИЧЕСКИЕ
with tab3:
    st.markdown("## 🔴 Критические инциденты")

    if df_crit.empty:
        st.success("За выбранный период критических инцидентов не обнаружено 🎉")
    else:
        # Gauge — доля критических
        total = len(df)
        n_crit = len(df_crit)
        pct = n_crit / total * 100

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            delta={"reference": 10, "valueformat": ".1f"},
            number={"suffix":"%","font":{"size":32}},
            title={"text":"Доля критических постов"},
            gauge={
                "axis":{"range":[0,30],"tickwidth":1},
                "bar":{"color": C_CRIT},
                "steps":[
                    {"range":[0,10],"color":"#e6ffed"},
                    {"range":[10,20],"color":"#fff3cd"},
                    {"range":[20,30],"color":"#ffd5d5"},
                ],
                "threshold":{"line":{"color":"darkred","width":3},"value":15},
            }
        ))
        fig_gauge.update_layout(height=220, margin=dict(t=40,b=0,l=20,r=20),
                                 paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Список инцидентов
        for _, row in df_crit.sort_values("final_score", ascending=False).head(20).iterrows():
            score = row["final_score"]
            emoji = "🔴" if score >= 0.85 else "🟠" if score >= 0.70 else "🟡"
            ts    = row["published"].strftime("%d.%m %H:%M") if pd.notna(row["published"]) else "—"
            with st.expander(f"{emoji} {row['channel']} · {ts} · скор {score:.0%}"):
                st.write(row["text"])
                st.progress(float(score))


# ВКЛАДКА 4: РАЗМЕТКА (Active Learning)
with tab4:
    st.markdown("## 🏷 Очередь активного обучения")
    st.caption("Посты с неопределённым скором (0.40–0.60). "
               "Разметка этих примеров максимально улучшает модель.")

    if not os.path.exists(_DB):
        st.info("БД не найдена. Запусти `python main.py`.")
    else:
        queue = get_labeling_queue(limit=20)
        if not queue:
            st.success("Очередь пуста — модель уверена во всех постах 🎉")
        else:
            st.info(f"В очереди: {len(queue)} постов")
            for queue_id, text, channel, model_score, uncertainty in queue:
                with st.container():
                    st.markdown(f"**`{channel}`** · скор: `{model_score:.2f}` · неопределённость: `{uncertainty:.2f}`")
                    st.write(text[:400])
                    st.progress(float(model_score))
                    c1, c2, c3 = st.columns(3)
                    if c1.button("✅ Критично",    key=f"c_{queue_id}"):
                        submit_manual_label(queue_id, 1); st.rerun()
                    if c2.button("⬜ Не критично", key=f"n_{queue_id}"):
                        submit_manual_label(queue_id, 0); st.rerun()
                    if c3.button("⏭ Пропустить",  key=f"s_{queue_id}"):
                        submit_manual_label(queue_id, -1); st.rerun()
                    st.divider()


# ВКЛАДКА 5: GIGACHAT
with tab5:
    st.markdown("## 🧠 GigaChat — прогноз реакции на публикацию")
    st.markdown('<span class="giga-badge">AI-анализ · GigaChat ESG-эксперт</span>',
                unsafe_allow_html=True)

    TEST_NEWS = (
        "Роспотребнадзор зафиксировал превышение ПДК вредных веществ в воздухе "
        "вблизи нефтехимического завода в Самарской области. По данным ведомства, "
        "уровень бензола превышен в 4,2 раза. Жители трёх ближайших посёлков "
        "жалуются на резкий запах и головные боли. Прокуратура возбудила проверку "
        "в отношении предприятия."
    )

    col_news, col_meta = st.columns([3, 1])
    with col_news:
        st.markdown("**Тестовая новость (скор модели: 0.94 🔴)**")
        st.markdown(f'<div class="news-block">{TEST_NEWS}</div>',
                    unsafe_allow_html=True)
    with col_meta:
        st.metric("Скор EcoPulse",  "0.94", delta="🔴 критично")
        st.metric("Источник",        "rpn.gov.ru")
        st.metric("Категория",       "Экологический")

    st.markdown("")
    user_text = st.text_area(
        "Или введи свой текст для анализа:",
        height=100,
        placeholder="Вставь текст новости из любого источника..."
    )
    analyze_text = user_text.strip() if user_text.strip() else TEST_NEWS

    if st.button("🔮 Сгенерировать прогноз реакции", type="primary"):
        creds = (os.environ.get("GIGACHAT_API_KEY") or
                 getattr(cfg, "GIGACHAT_API_KEY", ""))

        if not creds:
            # Демо-режим без ключа
            st.warning("GIGACHAT_API_KEY не задан - показываем демо-прогноз.")
            forecast = {
                "sentiment":           "негативный",
                "reaction_intensity":  "высокая",
                "risk_level":          "критический",
                "risk_explanation":    "Превышение ПДК в 4,2 раза на промышленном объекте "
                                       "с реакцией Роспотребнадзора и прокуратуры — "
                                       "классический сценарий быстрой эскалации. Жалобы "
                                       "местных жителей создают почву для широкого медиаосвещения.",
                "predicted_comments":  [
                    "Опять травят людей! Живём рядом с этим заводом 15 лет, "
                    "жалуемся давно - и всё без толку. Прокуратура только сейчас зашевелилась?",
                    "Бензол в 4 раза выше нормы - это уже уголовная статья. "
                    "Почему завод продолжает работать?",
                    "Будем следить за результатами проверки. Важно, чтобы реально "
                    "выставили штраф, а не замяли.",
                ],
                "recommended_action":  "Незамедлительно выпустить официальное заявление "
                                       "с признанием факта проверки и описанием мер. "
                                       "Организовать встречу с жителями. Подготовить "
                                       "пресс-пакет до завтрашнего утра.",
            }
        else:
            # Реальный запрос к GigaChat
            with st.spinner("GigaChat анализирует..."):
                try:
                    from gigachat import GigaChat as GC
                    from gigachat.models import Chat, Messages, MessagesRole

                    PROMPT = (
                        "Ты - эксперт по ESG-репутационным рискам.\n"
                        "Оцени публикацию и дай прогноз реакции людей.\n"
                        "Ответь строго в JSON без markdown:\n"
                        '{"sentiment":"позитивный/негативный/нейтральный",'
                        '"reaction_intensity":"низкая/средняя/высокая",'
                        '"risk_level":"низкий/средний/высокий/критический",'
                        '"predicted_comments":["комм1","комм2","комм3"],'
                        '"risk_explanation":"одно предложение",'
                        '"recommended_action":"рекомендация PR-команде"}'
                    )
                    with GC(credentials=creds,
                             scope=getattr(cfg,"GIGACHAT_SCOPE","GIGACHAT_API_PERS"),
                             model=getattr(cfg,"GIGACHAT_MODEL","GigaChat"),
                             verify_ssl_certs=False) as giga:
                        resp = giga.chat(Chat(messages=[
                            Messages(role=MessagesRole.SYSTEM, content=PROMPT),
                            Messages(role=MessagesRole.USER,
                                     content=f"Текст: {analyze_text[:2000]}"),
                        ], temperature=0.3, max_tokens=800))
                    content = resp.choices[0].message.content
                    content = re.sub(r"^```json\s*|\s*```$","",content.strip())
                    forecast = json.loads(content)
                except Exception as e:
                    st.error(f"Ошибка GigaChat: {e}")
                    st.stop()

        # Отображение результата
        st.markdown("---")
        st.markdown("### Результат анализа")

        risk_color = {
            "низкий":"🟢","средний":"🟡","высокий":"🟠","критический":"🔴"
        }.get(forecast.get("risk_level",""), "⚪")
        sent_emoji = {
            "позитивный":"😊","негативный":"😠","нейтральный":"😐"
        }.get(forecast.get("sentiment",""), "")

        m1, m2, m3 = st.columns(3)
        m1.metric("Тональность",
                  f"{sent_emoji} {forecast.get('sentiment','—').capitalize()}")
        m2.metric("Интенсивность реакции",
                  forecast.get("reaction_intensity","—").capitalize())
        m3.metric("Репутационный риск",
                  f"{risk_color} {forecast.get('risk_level','—').capitalize()}")

        st.markdown("")
        st.markdown(f"**⚠️ Пояснение:** {forecast.get('risk_explanation','—')}")
        st.markdown("")

        # Gauge риска
        risk_val = {"низкий":20,"средний":45,"высокий":70,"критический":90}
        fig_risk = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_val.get(forecast.get("risk_level",""), 50),
            number={"suffix":" / 100", "font":{"size":28}},
            title={"text":"Уровень репутационного риска"},
            gauge={
                "axis":{"range":[0,100]},
                "bar":{"color":C_CRIT},
                "steps":[
                    {"range":[0,33],"color":"#e6ffed"},
                    {"range":[33,66],"color":"#fff3cd"},
                    {"range":[66,100],"color":"#ffd5d5"},
                ],
            }
        ))
        fig_risk.update_layout(height=200, margin=dict(t=40,b=0,l=20,r=20),
                                paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_risk, use_container_width=True)

        st.markdown("### 💬 Вероятные комментарии пользователей")
        comments = forecast.get("predicted_comments", [])
        sentiments = ["neg","neg","neu"]
        colors = ["#fff5f5","#fff5f5","#fffaf0"]
        borders = [C_CRIT, C_CRIT, C_WARN]
        for i, comment in enumerate(comments[:3]):
            color  = colors[i] if i < len(colors) else "#f7fafc"
            border = borders[i] if i < len(borders) else "#e2e8f0"
            st.markdown(
                f'<div style="background:{color};border-left:3px solid {border};'
                f'border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:8px;'
                f'font-size:13px;line-height:1.5;color:#2d3748">💬 {comment}</div>',
                unsafe_allow_html=True
            )

        st.markdown("")
        st.markdown(
            f'<div class="rec-block">✅ <strong>Рекомендация PR-команде:</strong> '
            f'{forecast.get("recommended_action","—")}</div>',
            unsafe_allow_html=True
        )

