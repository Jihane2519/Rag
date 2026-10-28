import numpy as np
import re
from src.models import MinimalSource
from src.indexer import load_index, tokenize_code


def search(
    query: str,
    k: int = 5,
    tokenizer: str = "docs",
) -> list[MinimalSource]:
    bm25, chunks, texts = load_index(tokenizer)

    if tokenizer == "code":
        tokenized_query = tokenize_code(query)
    else:
        tokenized_query = re.findall(r"\w+", query.lower())

    scores = bm25.get_scores(tokenized_query)
    meilleur_k: list[int] = np.argsort(scores)[::-1][:k]

    return [chunks[i] for i in meilleur_k]
