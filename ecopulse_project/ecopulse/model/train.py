import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import Dataset



MODEL_NAME = "cointegrated/rubert-tiny2"

MAX_LENGTH = 64
BATCH_SIZE = 32
MAX_TRAIN_SAMPLES = 20000
NUM_EPOCHS = 3
LEARNING_RATE = 3e-5

TRAIN_PATH = "../data/data/train.csv"
VAL_PATH   = "../data/data/val.csv"
TEST_PATH  = "../data/data/test.csv"
OUTPUT_DIR = "model/ecopulse_rubert"

# Взвешенный Trainer
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(outputs.logits.device)
        )
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def make_hf_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
    ds = ds.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        ),
        batched=True,
        num_proc=1,
    )
    ds = ds.remove_columns(["text"])
    ds.set_format("torch")
    return ds


def train():
    # Загрузка данных
    print("[train] загружаем данные...")
    train_df = pd.read_csv(TRAIN_PATH).dropna(subset=["text", "label"])
    val_df   = pd.read_csv(VAL_PATH).dropna(subset=["text", "label"])

    train_df["label"] = train_df["label"].astype(int)
    val_df["label"]   = val_df["label"].astype(int)

    # Стратифицированная выборка для ускорения
    if MAX_TRAIN_SAMPLES and len(train_df) > MAX_TRAIN_SAMPLES:
        train_df, _ = train_test_split(
            train_df,
            train_size=MAX_TRAIN_SAMPLES,
            stratify=train_df["label"],
            random_state=42,
        )
        print(f"[train] выборка: {len(train_df)} строк (стратифицировано)")

    print(f"[train] train={len(train_df)}, val={len(val_df)}")
    print(f"[train] critical в train: {train_df['label'].mean():.1%}")

    # Токенизатор и датасеты
    print(f"[train] загружаем модель: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds  = make_hf_dataset(train_df, tokenizer)
    val_ds    = make_hf_dataset(val_df, tokenizer)

    # Веса классов
    weights = compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=train_df["label"].values
    )
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print(f"[train] class_weights = [{weights[0]:.2f}, {weights[1]:.2f}]")

    # Модель
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    # gradient checkpointing - экономит VRAM, чуть медленнее, но позволяет больший батч
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Определяем есть ли GPU
    has_gpu = torch.cuda.is_available()
    print(f"[train] GPU доступен: {has_gpu}")
    if has_gpu:
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}")

    # Аргументы обучения
    args = TrainingArguments(
        output_dir="model/checkpoints",

        # скорость
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        dataloader_num_workers=0,
        fp16=has_gpu,
        bf16=False,

        # обучение
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,

        # валидация и сохранение
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        # логи
        logging_steps=50,
        report_to="none",

        # воспроизводимость
        seed=42,
    )

    # Trainer
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Обучение
    print("\n[train] начинаем обучение...")
    print(f"[train] шагов на эпоху: {len(train_ds) // BATCH_SIZE}")
    print(f"[train] всего шагов: {len(train_ds) // BATCH_SIZE * NUM_EPOCHS}\n")

    trainer.train()

    # Оценка на тесте
    print("\n[train] оцениваем на test...")
    test_df = pd.read_csv(TEST_PATH).dropna(subset=["text", "label"])
    test_df["label"] = test_df["label"].astype(int)
    test_ds = make_hf_dataset(test_df, tokenizer)

    preds_out = trainer.predict(test_ds)
    y_pred = np.argmax(preds_out.predictions, axis=-1)
    y_true = test_df["label"].values

    print("\n=== Результат на test ===")
    print(classification_report(y_true, y_pred,
                                 target_names=["not_critical", "critical"]))

    # Сохранение
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n[train] модель сохранена → {OUTPUT_DIR}")

    return trainer


if __name__ == "__main__":
    train()
