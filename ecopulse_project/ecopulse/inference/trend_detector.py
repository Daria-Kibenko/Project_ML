import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

_model = None


def get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model


def detect_trending_clusters(posts: list, eps: float = 0.35, min_samples: int = 2):
    if len(posts) < min_samples:
        for p in posts:
            p["cluster_id"] = -1
            p["cluster_size"] = 1
        return posts

    embedder = get_embedder()
    texts = [p["text"] for p in posts]
    embeddings = embedder.encode(texts, normalize_embeddings=True)

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(embeddings)
    labels = clustering.labels_

    cluster_sizes = {}
    for label in labels:
        cluster_sizes[label] = cluster_sizes.get(label, 0) + 1

    for post, label in zip(posts, labels):
        post["cluster_id"] = int(label)
        post["cluster_size"] = cluster_sizes[label] if label != -1 else 1

    return posts


def trend_boost(cluster_size: int, max_boost: float = 0.15) -> float:
    if cluster_size <= 1:
        return 0.0
    boost = min(max_boost, (cluster_size - 1) * 0.04)
    return boost
