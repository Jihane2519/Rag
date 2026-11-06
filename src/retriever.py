import numpy as np
import re
from src.embedder import Embedder
from src.models import MinimalSource
from src.indexer import load_index, tokenize


# def search(
#     query: str,
#     k: int = 5,
#     tokenizer: str = "docs",
# ) -> list[MinimalSource]:
#     bm25, chunks, texts = load_index(tokenizer)

#     if tokenizer == "code":
#         tokenized_query = tokenize_code(query)
#     else:
#         tokenized_query = re.findall(r"\w+", query.lower())

#     scores = bm25.get_scores(tokenized_query)
#     meilleur_k: list[int] = np.argsort(scores)[::-1][:k]

#     return [chunks[i] for i in meilleur_k]

def search(
    query: str,
    k: int = 5,
) -> list[MinimalSource]:
    bm25, chunks, texts, embeddings = load_index()

    # if tokenizer == "code":
    #     tokenized_query = tokenize_code(query)
    # else:
    #     tokenized_query = re.findall(r"\w+", query.lower())
    embedder = Embedder()
    query_embedding = embedder.encode([query])[0]
    tokenized_query = tokenize(query)
    # tableaux NumPy (numpy.ndarray)
    scores = bm25.get_scores(tokenized_query)
    # tableaux NumPy (numpy.ndarray)
    similarity_scores = np.dot(embeddings, query_embedding)
    combined_scores = scores + similarity_scores
    meilleur_k: list[int] = np.argsort(combined_scores)[::-1][:k]

    return [chunks[i] for i in meilleur_k]



# import numpy as np
# import re
# from src.embedder import Embedder
# from src.models import MinimalSource
# from src.indexer import load_index, tokenize


# # def search(
# #     query: str,
# #     k: int = 5,
# #     tokenizer: str = "docs",
# # ) -> list[MinimalSource]:
# #     bm25, chunks, texts = load_index(tokenizer)

# #     if tokenizer == "code":
# #         tokenized_query = tokenize_code(query)
# #     else:
# #         tokenized_query = re.findall(r"\w+", query.lower())

# #     scores = bm25.get_scores(tokenized_query)
# #     meilleur_k: list[int] = np.argsort(scores)[::-1][:k]

# #     return [chunks[i] for i in meilleur_k]
# def normalize_scores(scores: np.ndarray) -> np.ndarray:
#     min_score = np.min(scores)
#     max_score = np.max(scores)

#     if max_score == min_score:
#         return np.zeros_like(scores)

#     return (scores - min_score) / (max_score - min_score)


# def search(
#     query: str,
#     k: int = 5,
# ) -> list[MinimalSource]:
#     bm25, chunks, texts, embeddings = load_index()

#     # if tokenizer == "code":
#     #     tokenized_query = tokenize_code(query)
#     # else:
#     #     tokenized_query = re.findall(r"\w+", query.lower())
#     embedder = Embedder()
#     query_embedding = embedder.encode([query])[0]
#     tokenized_query = tokenize(query)
#     # tableaux NumPy (numpy.ndarray)
#     scores = bm25.get_scores(tokenized_query)
#     # tableaux NumPy (numpy.ndarray)
#     similarity_scores = np.dot(embeddings, query_embedding)
#     # combined_scores = scores + similarity_scores
#     normalized_bm25 = normalize_scores(scores)
#     normalized_semantic = normalize_scores(similarity_scores)

#     combined_scores = (
#         0.5 * normalized_bm25
#         + 0.5 * normalized_semantic
#     )
#     meilleur_k: list[int] = np.argsort(combined_scores)[::-1][:k]

#     return [chunks[i] for i in meilleur_k]