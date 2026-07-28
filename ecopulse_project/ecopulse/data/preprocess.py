import random
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import nlpaug.augmenter.word as naw
    _HAS_NLPAUG = True
except ImportError:
    _HAS_NLPAUG = False


def augment_minority_class(df: pd.DataFrame, target_ratio: float = 0.3) -> pd.DataFrame:
    if not _HAS_NLPAUG:
        print("[augment] nlpaug не установлен, пропускаем аугментацию")
        return df

    aug = naw.SynonymAug(aug_src="wordnet", lang="rus")

    current_ratio = df["label"].mean()
    if current_ratio >= target_ratio:
        return df

    minority = df[df["label"] == 1]
    n_needed = int(target_ratio * len(df) / (1 - target_ratio)) - len(minority)
    n_needed = max(0, n_needed)

    print(f"[augment] текущая доля critical={current_ratio:.2%}, "
          f"генерируем {n_needed} новых примеров")

    new_rows = []
    pool = minority.sample(n=min(n_needed, len(minority) * 3),
                            replace=True, random_state=42)
    for _, row in pool.iterrows():
        try:
            aug_text = aug.augment(row["text"])[0]
            new_rows.append({**row.to_dict(), "text": aug_text, "source": row["source"] + "_aug"})
        except Exception:
            continue

    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


def synthetic_llm_generation_prompt() -> str:
    return """Напиши 10 постов в стиле Telegram-канала, которые описывают
реальный экологический или социальный инцидент на промышленном предприятии
в России. Стиль: короткий, эмоциональный, от лица очевидца или местного жителя.
Разные регионы, разные типы инцидентов (разлив, выброс, пожар, нарушение прав
работников, конфликт с местными жителями). Не используй один и тот же шаблон.
Выведи как пронумерованный список, без дополнительных пояснений."""


def time_based_split(df: pd.DataFrame, date_col: str = "date"):
    dated = df[df[date_col].notna()].sort_values(date_col)
    undated = df[df[date_col].isna()]

    n = len(dated)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    train_dated = dated.iloc[:train_end]
    val_dated = dated.iloc[train_end:val_end]
    test_dated = dated.iloc[val_end:]

    train_undated, temp_undated = train_test_split(
        undated, test_size=0.3, stratify=undated["label"], random_state=42
    )
    val_undated, test_undated = train_test_split(
        temp_undated, test_size=0.5, stratify=temp_undated["label"], random_state=42
    )

    train = pd.concat([train_dated, train_undated], ignore_index=True)
    val = pd.concat([val_dated, val_undated], ignore_index=True)
    test = pd.concat([test_dated, test_undated], ignore_index=True)

    train = train.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"[split] train={len(train)} val={len(val)} test={len(test)}")
    print(f"[split] доля critical: train={train['label'].mean():.2%} "
          f"val={val['label'].mean():.2%} test={test['label'].mean():.2%}")

    return train, val, test


if __name__ == "__main__":
    df = pd.read_csv("data/dataset_combined.csv", parse_dates=["date"])
    df = augment_minority_class(df, target_ratio=0.3)
    train, val, test = time_based_split(df)

    train.to_csv("data/train.csv", index=False)
    val.to_csv("data/val.csv", index=False)
    test.to_csv("data/test.csv", index=False)
