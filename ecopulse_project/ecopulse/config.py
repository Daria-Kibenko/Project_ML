from dataclasses import dataclass, field


@dataclass
class Config:
    # Model
    MODEL_NAME: str = "ai-forever/ruBert-base"
    ONNX_PATH: str = "model/ecopulse.onnx"
    CALIBRATION_PATH: str = "model/temperature.json"
    MAX_LENGTH: int = 128
    THRESHOLD: float = 0.65

    UNCERTAINTY_LOW: float = 0.40
    UNCERTAINTY_HIGH: float = 0.60

    ENSEMBLE_WEIGHTS: dict = field(default_factory=lambda: {
        "rubert": 0.6,
        "sbert": 0.4,
    })

    # Source scoring
    SCORE_MODEL_WEIGHT: float = 0.7
    SCORE_SOURCE_WEIGHT: float = 0.3

    # Bot & API
    VK_TOKEN: str = "..."
    BOT_TOKEN: str = "..."
    ANALYST_CHAT_ID: int = ...
    DIGEST_HOUR: int = 8

    # GigaChat
    GIGACHAT_API_KEY: str = "..."
    GIGACHAT_SCOPE: str = "GIGACHAT_API_PERS"
    GIGACHAT_MODEL: str = "GigaChat"

    # Parser
    PARSE_INTERVAL_MINUTES: int = 15
    MAX_SUBSCRIBERS_FOR_NORM: int = 50000
    SESSION_NAME: str = "ecopulse"

    # Retrain
    RETRAIN_FEEDBACK_THRESHOLD: int = 200

    # DB
    DB_PATH: str = "db/ecopulse.db"

    # Drift monitoring
    DRIFT_CHECK_INTERVAL_DAYS: int = 7
    REFERENCE_DATA_PATH: str = "data/train_reference.csv"


cfg = Config()
