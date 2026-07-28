import re
import pandas as pd
from datasets import load_dataset


def clean_text(text) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


# 1. Banki.ru отзывы
def load_bank_reviews() -> pd.DataFrame:
    ds = load_dataset("Romjiik/Russian_bank_reviews", split="train")
    df = ds.to_pandas()

    out = pd.DataFrame()
    out["text"] = df["review"].apply(clean_text)
    rating = pd.to_numeric(df["rating_value"], errors="coerce")
    out["label"] = (rating <= 2).astype(int)
    out = out[rating != 3]                     # пограничный рейтинг - выкидываем
    out["source"] = "bank_reviews"
    out["date"] = pd.to_datetime(df["review_dttm"], errors="coerce")

    return out[["text", "label", "source", "date"]]


# 2. Токсичные комментарии
def load_toxic_comments() -> pd.DataFrame:
    ds = load_dataset("AlexSham/Toxic_Russian_Comments", split="train")
    df = ds.to_pandas()

    print("[toxic_comments] уникальные значения label:", df["label"].unique())

    THREAT_LABEL_ID = 2

    out = pd.DataFrame()
    out["text"] = df["text"].apply(clean_text)
    out["label"] = (df["label"] == THREAT_LABEL_ID).astype(int)
    out["source"] = "toxic_comments"
    out["date"] = pd.NaT

    return out[["text", "label", "source", "date"]]


# 3. Яндекс.Карты отзывы (главный новый источник)
def load_geo_reviews(sample_n: int = 50000) -> pd.DataFrame:
    ds = load_dataset("d0rj/geo-reviews-dataset-2023", split="train")
    df = ds.to_pandas()
    df = df.sample(n=min(sample_n, len(df)), random_state=42)

    out = pd.DataFrame()
    out["text"] = df["text"].apply(clean_text)
    rating = pd.to_numeric(df["rating"], errors="coerce")
    out["label"] = (rating <= 2).astype(int)
    out = out[rating != 3]
    out["source"] = "geo_reviews"
    out["date"] = pd.NaT

    return out[["text", "label", "source", "date"]]


# Очистка и сборка
def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    df = df[df["text"].str.len() > 20]
    print(f"[dedup] {before} → {len(df)} строк после очистки")
    return df


def build_dataset(save_path: str = "data/dataset_combined.csv") -> pd.DataFrame:
    parts = [
        load_bank_reviews(),
        load_toxic_comments(),
        load_geo_reviews(),
    ]
    df = pd.concat(parts, ignore_index=True)
    df = deduplicate(df)

    print("\n[распределение классов]")
    print(df["label"].value_counts(normalize=True))
    print("\n[распределение по источникам]")
    print(df.groupby(["source", "label"]).size())

    df.to_csv(save_path, index=False)
    print(f"\nСохранено: {save_path} ({len(df)} строк)")
    return df


if __name__ == "__main__":
    build_dataset()
