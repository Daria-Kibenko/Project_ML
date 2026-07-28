import os
import sqlite3
from contextlib import contextmanager

from ecopulse.config import cfg

_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DB_PATH = os.path.join(_ROOT, "ecopulse", cfg.DB_PATH)
os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)


@contextmanager
def get_con():
    con = sqlite3.connect(_DB_PATH)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with get_con() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS posts (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                channel       TEXT,
                message_id    TEXT,
                text          TEXT,
                published     DATETIME,
                parsed_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_score   REAL,
                source_weight REAL,
                final_score   REAL,
                label         INTEGER,
                notified      INTEGER DEFAULT 0,
                UNIQUE(channel, message_id)
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id     INTEGER REFERENCES posts(id),
                user_id     INTEGER,
                is_correct  INTEGER,
                created     DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS labeling_queue (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id      INTEGER REFERENCES posts(id),
                uncertainty  REAL,
                status       TEXT DEFAULT 'pending',
                manual_label INTEGER,
                added        DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS retrain_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                triggered_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
                n_new_samples INTEGER,
                f1_before     REAL,
                f1_after      REAL
            );
        """)


def save_post(channel, message_id, text, published,
              model_score, source_weight, label):
    final_score = (
        model_score * cfg.SCORE_MODEL_WEIGHT
        + source_weight * cfg.SCORE_SOURCE_WEIGHT
    )
    with get_con() as con:
        con.execute("""
            INSERT OR IGNORE INTO posts
              (channel, message_id, text, published,
               model_score, source_weight, final_score, label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (channel, str(message_id), text, published,
              model_score, source_weight, final_score, int(label)))

        row = con.execute(
            "SELECT id FROM posts WHERE channel=? AND message_id=?",
            (channel, str(message_id))
        ).fetchone()
        post_id = row[0] if row else None

        if post_id and cfg.UNCERTAINTY_LOW <= model_score <= cfg.UNCERTAINTY_HIGH:
            con.execute(
                "INSERT INTO labeling_queue (post_id, uncertainty) VALUES (?, ?)",
                (post_id, abs(model_score - 0.5))
            )
        return post_id


def save_feedback(post_id, user_id, is_correct):
    with get_con() as con:
        con.execute(
            "INSERT INTO feedback (post_id, user_id, is_correct) VALUES (?, ?, ?)",
            (post_id, user_id, int(is_correct))
        )


def get_todays_critical(limit=5):
    with get_con() as con:
        return con.execute("""
            SELECT id, channel, text, published, final_score
            FROM posts
            WHERE label=1 AND date(parsed_at)=date('now')
            ORDER BY final_score DESC LIMIT ?
        """, (limit,)).fetchall()


def mark_notified(post_id):
    with get_con() as con:
        con.execute("UPDATE posts SET notified=1 WHERE id=?", (post_id,))


def get_labeling_queue(limit=50):
    with get_con() as con:
        return con.execute("""
            SELECT lq.id, p.text, p.channel, p.model_score, lq.uncertainty
            FROM labeling_queue lq
            JOIN posts p ON p.id = lq.post_id
            WHERE lq.status='pending'
            ORDER BY lq.uncertainty ASC LIMIT ?
        """, (limit,)).fetchall()


def submit_manual_label(queue_id, label):
    with get_con() as con:
        con.execute(
            "UPDATE labeling_queue SET status='labeled', manual_label=? WHERE id=?",
            (label, queue_id)
        )


def count_new_feedback_since_last_retrain():
    with get_con() as con:
        last = con.execute(
            "SELECT MAX(triggered_at) FROM retrain_log"
        ).fetchone()[0]
        if last:
            row = con.execute(
                "SELECT COUNT(*) FROM feedback WHERE created > ?", (last,)
            ).fetchone()
        else:
            row = con.execute("SELECT COUNT(*) FROM feedback").fetchone()
        return row[0]


def log_retrain(n_samples, f1_before, f1_after):
    with get_con() as con:
        con.execute(
            "INSERT INTO retrain_log (n_new_samples, f1_before, f1_after) VALUES (?, ?, ?)",
            (n_samples, f1_before, f1_after)
        )
