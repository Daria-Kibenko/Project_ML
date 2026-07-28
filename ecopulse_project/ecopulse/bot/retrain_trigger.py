import os
import logging
import pandas as pd

from  ecopulse.config import cfg
from  ecopulse.db.storage import count_new_feedback_since_last_retrain, log_retrain, get_con

logger = logging.getLogger(__name__)


def fetch_new_labeled_data():
    with get_con() as con:
        df = pd.read_sql("""
            SELECT p.text, f.is_correct, p.label AS model_label
            FROM feedback f JOIN posts p ON p.id = f.post_id
        """, con)
    df["label"] = df.apply(
        lambda r: r["model_label"] if r["is_correct"] == 1 else 1 - r["model_label"],
        axis=1
    )
    return df[["text", "label"]]


def check_and_retrain():
    n_new = count_new_feedback_since_last_retrain()
    logger.info(f"[retrain] новых фидбеков: {n_new}")
    if n_new < cfg.RETRAIN_FEEDBACK_THRESHOLD:
        return
    logger.info("[retrain] запускаем дообучение...")
    try:
        from  ecopulse.model.evaluate import evaluate_model
        from  ecopulse.model.train import train_model
        from  ecopulse.model.export import export_to_onnx
        import torch

        f1_before = evaluate_model().get("f1", 0)
        new_data  = fetch_new_labeled_data()

        _ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        train_path = os.path.join(_ROOT, "ecopulse", "data", "train.csv")
        train_df   = pd.read_csv(train_path)
        pd.concat([train_df, new_data]).to_csv(train_path, index=False)

        train_model()
        export_to_onnx(torch.device("cpu"))

        f1_after = evaluate_model().get("f1", 0)
        log_retrain(n_new, f1_before, f1_after)
        logger.info(f"[retrain] F1: {f1_before:.3f} → {f1_after:.3f}")
    except Exception as e:
        logger.error(f"[retrain] ошибка: {e}")
