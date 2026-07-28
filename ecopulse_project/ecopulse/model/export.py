import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

THIS_DIR          = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR          = os.path.dirname(THIS_DIR)
MODEL_DIR         = os.path.join(ROOT_DIR, "model/model", "ecopulse_rubert")
ONNX_PATH         = os.path.join(ROOT_DIR, "model/model", "ecopulse.onnx")
CALIBRATION_PATH  = os.path.join(ROOT_DIR, "model/model", "temperature.json")

MAX_LENGTH = 64


def export_to_onnx(device: torch.device):
    print(f"\n[export] загружаем модель из {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    # Фиктивный ввод для трассировки вычислительного графа
    dummy_text = "произошёл разлив химикатов на заводе, жители эвакуированы"
    enc = tokenizer(
        dummy_text,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    print(f"[export] экспортируем в ONNX → {ONNX_PATH}...")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        ONNX_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits":         {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,   # оптимизация: свёртка констант
    )
    print(f"[export] ✅ ONNX сохранён ({os.path.getsize(ONNX_PATH) / 1e6:.1f} MB)")
    return model, tokenizer, enc


def verify_onnx(model, tokenizer, enc, device: torch.device):
    """Проверяет, что ONNX выдаёт те же результаты что и PyTorch."""
    print("\n[verify] проверяем совпадение ONNX и PyTorch...")

    import onnxruntime as ort

    # Выбираем провайдер: GPU если доступен, иначе CPU
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device.type == "cuda"
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(ONNX_PATH, providers=providers)

    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # PyTorch предсказание
    with torch.no_grad():
        torch_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).logits.cpu().numpy()

    # ONNX предсказание
    onnx_logits = session.run(None, {
        "input_ids":      enc["input_ids"].numpy().astype(np.int64),
        "attention_mask": enc["attention_mask"].numpy().astype(np.int64),
    })[0]

    max_diff = np.abs(torch_logits - onnx_logits).max()
    print(f"[verify] макс. расхождение: {max_diff:.8f}")

    if max_diff < 1e-3:
        print("[verify] ✅ ONNX корректен — результаты совпадают")
    else:
        print("[verify] ⚠️  Расхождение выше порога — проверь экспорт")

    return session


def benchmark_speed(session, tokenizer):
    print("\n[benchmark] замеряем скорость инференса на CPU...")

    texts = [
        "на заводе произошёл пожар, есть пострадавшие",
        "отличное обслуживание, всё понравилось",
        "банк заблокировал карту без предупреждения, деньги списали",
        "разлив нефти в реке Обь, местные жители в панике",
        "хорошая кофейня, советую попробовать круассаны",
    ]

    enc = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="np",
    )
    inputs = {
        "input_ids":      enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }

    # прогрев
    for _ in range(3):
        session.run(None, inputs)

    # замер
    N = 50
    start = time.perf_counter()
    for _ in range(N):
        session.run(None, inputs)
    elapsed = time.perf_counter() - start

    per_batch = elapsed / N * 1000
    per_text  = per_batch / len(texts)
    print(f"[benchmark] батч {len(texts)} текстов: {per_batch:.1f} мс")
    print(f"[benchmark] один текст:               {per_text:.1f} мс")
    print(f"[benchmark] пропускная способность:   {1000 / per_text:.0f} текстов/сек")


def calibrate_temperature(model, tokenizer, device: torch.device):
    import pandas as pd
    from torch.utils.data import DataLoader, TensorDataset

    val_path = os.path.join(ROOT_DIR, "data", "val.csv")
    if not os.path.exists(val_path):
        print("[calibrate] val.csv не найден — пропускаем калибровку")
        return 1.0

    print("\n[calibrate] подбираем температуру на val.csv...")
    df = pd.read_csv(val_path).dropna(subset=["text", "label"]).head(2000)
    df["label"] = df["label"].astype(int)

    enc = tokenizer(
        list(df["text"]),
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    labels_tensor = torch.tensor(df["label"].values, dtype=torch.long)

    dataset = TensorDataset(
        enc["input_ids"], enc["attention_mask"], labels_tensor
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    # Собираем логиты
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for ids, mask, lbls in loader:
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device))
            all_logits.append(out.logits.cpu())
            all_labels.append(lbls)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    # Оптимизация температуры
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer   = torch.optim.LBFGS([temperature], lr=0.01, max_iter=100)
    loss_fn     = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    t_value = float(temperature.item())
    print(f"[calibrate] подобранная температура T = {t_value:.4f}")

    with open(CALIBRATION_PATH, "w") as f:
        json.dump({"temperature": t_value}, f)
    print(f"[calibrate] ✅ сохранено → {CALIBRATION_PATH}")
    return t_value


def main():
    # Устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[export] устройство: {device}")
    if device.type == "cuda":
        print(f"[export] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[export] GPU не найден — экспортируем на CPU (это нормально)")

    # Проверка модели
    if not os.path.exists(MODEL_DIR):
        print(f"\n❌ Не найдена модель: {MODEL_DIR}")
        print("   Сначала запусти: python model/train.py")
        sys.exit(1)

    os.makedirs(os.path.join(ROOT_DIR, "model"), exist_ok=True)

    # Экспорт
    model, tokenizer, enc = export_to_onnx(device)

    # Верификация
    try:
        import onnxruntime
        session = verify_onnx(model, tokenizer, enc, device)
        benchmark_speed(session, tokenizer)
    except ImportError:
        print("[verify] onnxruntime не установлен — пропускаем верификацию")
        print("         pip install onnxruntime")

    # Калибровка температуры
    calibrate_temperature(model, tokenizer, device)

    print(f"\n✅ Экспорт завершён:")
    print(f"   ONNX-модель:  {ONNX_PATH}")
    print(f"   Калибровка:   {CALIBRATION_PATH}")


if __name__ == "__main__":
    main()
