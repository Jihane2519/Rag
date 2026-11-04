import os
import pickle
import re
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from src.models import MinimalSource
from src.config import DATA_RAW_PATH, DATA_PROCESSED_PATH, DEFAULT_CHUNK_SIZE
from src.loader import load_files
from src.chunker import chunk_file


def tokenize(text: str) -> list[str]:
    """Tokenize identifiers and split their parts."""
    tokens = re.findall(r"\w+", text.lower())
    result = []

    for token in tokens:
        result.append(token)

        # get_user_data -> get, user, data
        parts = re.split(r"[_\d]+", token)
        result.extend(part for part in parts if part and part != token)

    return result


def build_index(max_chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
    files = load_files(DATA_RAW_PATH)
    all_chunks: list[MinimalSource] = []
    # pour le texte entre start et end
    all_texts: list[str] = []

    for file_path, content in tqdm(files, desc="Chunking"):
        chunks = chunk_file(file_path, content, max_chunk_size)
        for chunk in chunks:
            all_chunks.append(chunk)
            # extraire le texte du chunk
            chunk_text = content[
                chunk.first_character_index:chunk.last_character_index
            ]
            all_texts.append(chunk_text)
    print("Building BM25 index...")
    # ex: [["python", "is", "powerful"]]
    # Chaque liste représente un document/chunk.
    # tokenized_docs = [text.lower().split() for text in all_texts]
    # tokenized_docs = [
    #     re.findall(r'\w+', text.lower())
    #     for text in tqdm(all_texts, desc="Tokenizing")
    # ]
    tokenized = [
        tokenize(text)
        for text in tqdm(all_texts, desc="Tokenizing")
    ]
    bm25 = BM25Okapi(tokenized)
    # bm25_docs = BM25Okapi(tokenized_docs)
    # bm25_code = BM25Okapi(tokenized_code)
    # print(type(bm25)) -> <class 'rank_bm25.BM25Okapi'>

    os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)

    # with open(f"{DATA_PROCESSED_PATH}/bm25_docs.pkl", "wb") as f:
    #     pickle.dump(bm25_docs, f)
    # with open(f"{DATA_PROCESSED_PATH}/bm25_code.pkl", "wb") as f:
    #     pickle.dump(bm25_code, f)
    with open(f"{DATA_PROCESSED_PATH}/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(f"{DATA_PROCESSED_PATH}/chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(f"{DATA_PROCESSED_PATH}/texts.pkl", "wb") as f:
        pickle.dump(all_texts, f)

    print(
        f"Ingestion complete! Indexed {len(all_chunks)}"
        "chunks under {DATA_PROCESSED_PATH}/"
    )


# def load_index(
#     tokenizer: str = "docs",
# ) -> tuple[BM25Okapi, list[MinimalSource], list[str]]:
#     if tokenizer == "code":
#         index_path = f"{DATA_PROCESSED_PATH}/bm25_code.pkl"
#     else:
#         index_path = f"{DATA_PROCESSED_PATH}/bm25_docs.pkl"

#     with open(index_path, "rb") as f:
#         bm25 = pickle.load(f)

#     with open(f"{DATA_PROCESSED_PATH}/chunks.pkl", "rb") as f:
#         chunks: list[MinimalSource] = pickle.load(f)

#     with open(f"{DATA_PROCESSED_PATH}/texts.pkl", "rb") as f:
#         texts: list[str] = pickle.load(f)

#     return bm25, chunks, texts


def load_index() -> tuple[BM25Okapi, list[MinimalSource], list[str]]:
    # if tokenizer == "code":
    #     index_path = f"{DATA_PROCESSED_PATH}/bm25_code.pkl"
    # else:
    #     index_path = f"{DATA_PROCESSED_PATH}/bm25_docs.pkl"
    index_path = f"{DATA_PROCESSED_PATH}/bm25.pkl"
    with open(index_path, "rb") as f:
        bm25 = pickle.load(f)

    with open(f"{DATA_PROCESSED_PATH}/chunks.pkl", "rb") as f:
        chunks: list[MinimalSource] = pickle.load(f)

    with open(f"{DATA_PROCESSED_PATH}/texts.pkl", "rb") as f:
        texts: list[str] = pickle.load(f)

    return bm25, chunks, texts
