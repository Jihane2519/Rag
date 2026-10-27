import numpy as np
import re
from src.models import MinimalSource
from src.indexer import load_index


def search(query: str, k: int = 5) -> list[MinimalSource]:
    """
    Cherche les k chunks les plus pertinents pour une question.

    Args:
        query: la question
        k: nombre de résultats à retourner

    Returns:
        liste de MinimalSource (top-k)
    """
    # charger l'index BM25, les chunks et les textes
    bm25, chunks, texts = load_index()

    # tokenizer la query comme les chunks
    # tokenized_query = query.lower().split()
    tokenized_query = re.findall(r'\b\w+\b', query.lower())

    # calculer les scores BM25 pour chaque chunk
    scores = bm25.get_scores(tokenized_query)

    # C'est le coeur du retriever.
    # ex scores = [1.2, 0.0, 4.8, 2.1, 0.5]
    meilleur_k: list[int] = np.argsort(scores)[::-1][:k]

    # retourner les chunks correspondants
    return [chunks[i] for i in meilleur_k]
