import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, average_precision_score,
    roc_auc_score
)
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Пути
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(THIS_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, "data/data")
MODEL_DIR  = os.path.join(ROOT_DIR, "model/model", "ecopulse_rubert")
TEST_PATH  = os.path.join(DATA_DIR, "test.csv")

# Настройки
MAX_LENGTH  = 64
BATCH_SIZE  = 64    # на GPU можно больше; на CPU уменьши до 16
THRESHOLD   = 0.5   # порог для класса critical


class TextDataset(TorchDataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label":          self.labels[idx],
        }


def evaluate():
    # Устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] устройство: {device}")
    if device.type == "cuda":
        print(f"[evaluate] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[evaluate] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Проверка файлов
    for path, name in [(TEST_PATH, "test.csv"), (MODEL_DIR, "model/ecopulse_rubert")]:
        if not os.path.exists(path):
            print(f"\n❌ Не найдено: {path}")
            print(f"   Сначала запусти: python model/train.py")
            sys.exit(1)

    # Загрузка данных
    print(f"\n[evaluate] загружаем test.csv...")
    df = pd.read_csv(TEST_PATH).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    print(f"[evaluate] строк: {len(df)}, critical: {df['label'].mean():.1%}")

    # Загрузка модели
    print(f"[evaluate] загружаем модель из {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    # Инференс батчами на GPU/CPU
    print(f"[evaluate] инференс (batch_size={BATCH_SIZE})...")
    dataset = TextDataset(df["text"], df["label"], tokenizer, MAX_LENGTH)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=(device.type == "cuda"))

    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=-1)[:, 1]  # P(critical)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

            if (i + 1) % 10 == 0:
                print(f"[evaluate] батч {i+1}/{len(loader)}")

    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs >= THRESHOLD).astype(int)

    # Основные метрики
    print("\n" + "="*50)
    print("МЕТРИКИ НА TEST-ВЫБОРКЕ")
    print("="*50)
    print(classification_report(
        labels, preds,
        target_names=["not_critical", "critical"],
        digits=4
    ))

    print("Confusion Matrix:")
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")
    print()

    pr_auc = average_precision_score(labels, probs)
    roc_auc = roc_auc_score(labels, probs)
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Разбивка по источникам
    if "source" in df.columns:
        print("\n" + "="*50)
        print("МЕТРИКИ ПО ИСТОЧНИКАМ")
        print("="*50)
        df["pred"]  = preds
        df["prob"]  = probs

        for source, group in df.groupby("source"):
            if len(group) < 10:
                continue
            p, r, f1, _ = precision_recall_fscore_support(
                group["label"], group["pred"],
                average="binary", zero_division=0
            )
            n_crit = group["label"].sum()
            print(f"{source:<30}  n={len(group):5d}  critical={n_crit:4d}"
                  f"  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")

    # Анализ пороговых значений
    print("\n" + "="*50)
    print("ПОДБОР ПОРОГА (Precision vs Recall)")
    print("="*50)
    print(f"{'Порог':>6}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}  {'N alerts':>8}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8]:
        p_t = (probs >= t).astype(int)
        if p_t.sum() == 0:
            continue
        pr, rc, f1, _ = precision_recall_fscore_support(
            labels, p_t, average="binary", zero_division=0
        )
        print(f"{t:>6.2f}  {pr:>9.3f}  {rc:>6.3f}  {f1:>6.3f}  {p_t.sum():>8}")

    print("\n✅ Оценка завершена")
    return {"pr_auc": pr_auc, "roc_auc": roc_auc}


if __name__ == "__main__":
    evaluate()
