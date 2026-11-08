import os
import pickle
import re
import numpy as np
import hashlib
import json
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from src.models import MinimalSource
from src.config import DATA_RAW_PATH, DATA_PROCESSED_PATH, DEFAULT_CHUNK_SIZE
from src.loader import load_files
from src.chunker import chunk_file
from src.embedder import Embedder
from pathlib import Path


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


def get_file_hash(content: str) -> str:
    """Calculate the SHA-256 hash of a file's content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def save_manifest(
    manifest: dict[str, str],
    config: dict[str, int]
) -> None:
    """Save current file hashes and indexing configuration."""
    manifest_path = f"{DATA_PROCESSED_PATH}/manifest.json"
    data = {
        "config": config,
        "files": manifest
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# Cette fonction récupère les anciens hashes.
def load_manifest() -> dict:
    """Load previously saved file hashes."""

    manifest_path = f"{DATA_PROCESSED_PATH}/manifest.json"
    if not os.path.exists(manifest_path):
        return {
            "config": {},
            "files": {}
        }
        # Nous n'avons donc aucun ancien hash à récupérer(première indexation)
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # lit le contenu JSON et le transforme en dictionnaire Python
    if "files" not in data:
        return {"config": {}, "files": data}
    return data


def detect_changes(
    files: list[tuple[str, str]],
    old_manifest: dict,
) -> tuple[list[tuple[str, str]], list[str], dict[str, str]]:
    new_manifest = {}
    changed_files = []
    old_files = old_manifest["files"]
    for file_path, content in files:
        file_hash = get_file_hash(content)
        new_manifest[file_path] = file_hash
        # in case file not exist in old_manifest(new_file) get return None
        if old_files.get(file_path) != file_hash:
            # Nouveaux fichiers et fichiers modifiés
            changed_files.append((file_path, content))
    deleted_files = [
        path for path in old_files
        if path not in new_manifest
    ]
    return changed_files, deleted_files, new_manifest


def build_index(max_chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
    files = load_files(DATA_RAW_PATH)
    all_chunks: list[MinimalSource] = []
    # pour le texte entre start et end
    all_texts: list[str] = []
    # le text chnger pour le donner a encode
    new_texts: list[str] = []
    all_embeddings = []
    old_data = {}
    old_manifest = load_manifest()
    current_config = {
        "max_chunk_size": max_chunk_size,
        "overlap": 200
    }
    config_changed = old_manifest["config"] != current_config
    changed_files, deleted_files, new_manifest = detect_changes(
        files,
        old_manifest,
    )
    if config_changed:
        print("Chunker configuration changed. Re-indexing all files ...")
        changed_files = files
    if (Path(DATA_PROCESSED_PATH) / "chunks.pkl").exists():
        _, old_chunks, old_texts, old_embeddings = load_index()
    else:
        old_chunks = []
        old_texts = []
        old_embeddings = np.empty((0, 384))

    for chunk, text, embedding in zip(
        old_chunks,
        old_texts,
        old_embeddings
    ):
        old_data.setdefault(chunk.file_path, []).append(
            (chunk, text, embedding)
        )
    # solution car on ne peut pas dire file path int (file_path, content)
    # ou bien on va returner seulement les files si ona pas utiliser les contents
    changed_paths = {file_path for file_path, content in changed_files}
    for file_path, data in old_data.items():
        if file_path not in changed_paths and file_path not in deleted_files:
            for chunk, text, embedding in data:
                all_chunks.append(chunk)
                all_texts.append(text)
                all_embeddings.append(embedding)
    for file_path, content in tqdm(changed_files, desc="Chunking"):
        chunks = chunk_file(file_path, content, max_chunk_size)
        for chunk in chunks:
            all_chunks.append(chunk)
            # extraire le texte du chunk
            chunk_text = content[
                chunk.first_character_index:chunk.last_character_index
            ]
            all_texts.append(chunk_text)
            new_texts.append(chunk_text)
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
    if new_texts:
        embedder = Embedder()
        new_embeddings = embedder.encode(new_texts)
        all_embeddings.extend(new_embeddings)
    # car all_embeddings est une list []
    embeddings = np.array(all_embeddings)
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
    with open(f"{DATA_PROCESSED_PATH}/embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    save_manifest(new_manifest, current_config)

    print(
        f"Ingestion complete! Indexed {len(all_chunks)} "
        f"chunks under {DATA_PROCESSED_PATH}/"
    )
    print("Chunks:", len(all_chunks))
    print("Texts:", len(all_texts))
    print("Embeddings:", len(all_embeddings))
    assert len(all_chunks) == len(all_texts) == len(all_embeddings), (
        "Mismatch between chunks, texts and embeddings!"
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


def load_index() -> tuple[
    BM25Okapi,
    list[MinimalSource],
    list[str],
    np.ndarray
]:
    # if tokenizer == "code":
    #     index_path = f"{DATA_PROCESSED_PATH}/bm25_code.pkl"
    # else:
    #     index_path = f"{DATA_PROCESSED_PATH}/bm25_docs.pkl"
    # index_path = f"{DATA_PROCESSED_PATH}/bm25.pkl"
    with open(f"{DATA_PROCESSED_PATH}/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    with open(f"{DATA_PROCESSED_PATH}/chunks.pkl", "rb") as f:
        chunks: list[MinimalSource] = pickle.load(f)

    with open(f"{DATA_PROCESSED_PATH}/texts.pkl", "rb") as f:
        texts: list[str] = pickle.load(f)

    with open(f"{DATA_PROCESSED_PATH}/embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    return bm25, chunks, texts, embeddings
