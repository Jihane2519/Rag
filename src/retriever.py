import numpy as np
from rank_bm25 import BM25Okapi

from src.embedder import Embedder
from src.models import MinimalSource
from src.indexer import load_index, tokenize
from src.cache import _query_cache


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    min_score = np.min(scores)
    max_score = np.max(scores)

    if max_score == min_score:
        return np.zeros_like(scores)

    return np.asarray(
        (scores - min_score) / (max_score - min_score)
    )


def search_re(
    query: str,
    k: int = 5,
    embedder: Embedder | None = None,
    index: tuple[
        BM25Okapi,
        list[MinimalSource],
        list[str],
        np.ndarray
    ] | None = None,
) -> list[MinimalSource]:
    # global _query_cache
    if (query, k) in _query_cache:
        # print("\nCACHE HIT")
        return _query_cache[(query, k)]

    if index is None:
        index = load_index()

    if embedder is None:
        embedder = Embedder()

    bm25, chunks, texts, embeddings = index

    query_embedding = embedder.encode([query])[0]
    tokenized_query = tokenize(query)

    # tableaux NumPy (numpy.ndarray)
    scores = bm25.get_scores(tokenized_query)

    # tableaux NumPy (numpy.ndarray)
    similarity_scores = np.dot(embeddings, query_embedding)

    # combined_scores = scores + similarity_scores
    normalized_bm25 = normalize_scores(scores)
    normalized_semantic = normalize_scores(similarity_scores)

    combined_scores = (
        0.5 * normalized_bm25
        + 0.5 * normalized_semantic
    )

    meilleur_k: np.ndarray = np.argsort(combined_scores)[::-1][:k]

    _query_cache[(query, k)] = [chunks[i] for i in meilleur_k]

    # print("\nCACHE MISS")
    return _query_cache[(query, k)]
