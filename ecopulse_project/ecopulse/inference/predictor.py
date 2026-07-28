import json
import numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort
from ecopulse.config import cfg


def softmax(x, temperature=1.0):
    x = x / temperature
    e = np.exp(x - np.max(x))
    return e / e.sum()


class Predictor:
    def __init__(self, onnx_path: str = None, calibration_path: str = None):
        onnx_path = onnx_path or cfg.ONNX_PATH
        calibration_path = calibration_path or cfg.CALIBRATION_PATH

        self.tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruBert-base")
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )

        try:
            with open(calibration_path) as f:
                self.temperature = json.load(f)["temperature"]
        except FileNotFoundError:
            self.temperature = 1.0  # без калибровки, если файла нет

    def _raw_score(self, text: str) -> float:
        enc = self.tokenizer(
            text, max_length=cfg.MAX_LENGTH, padding="max_length",
            truncation=True, return_tensors="np"
        )
        inputs = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        logits = self.session.run(None, inputs)[0][0]
        probs = softmax(logits, temperature=self.temperature)
        return float(probs[1])

    def predict(self, text: str, source_weight: float = 0.5) -> dict:
        model_score = self._raw_score(text)
        final_score = (
            model_score * cfg.SCORE_MODEL_WEIGHT
            + source_weight * cfg.SCORE_SOURCE_WEIGHT
        )
        label = int(final_score >= cfg.THRESHOLD)
        return {
            "model_score": model_score,
            "source_weight": source_weight,
            "final_score": final_score,
            "label": label,
        }

    def predict_batch(self, texts, source_weights=None):
        source_weights = source_weights or [0.5] * len(texts)
        return [self.predict(t, w) for t, w in zip(texts, source_weights)]


def source_weight(subscribers: int) -> float:
    if subscribers <= 0:
        return 0.0
    norm = np.log(subscribers + 1) / np.log(cfg.MAX_SUBSCRIBERS_FOR_NORM + 1)
    return float(min(1.0, norm))
