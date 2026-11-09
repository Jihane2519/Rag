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
    max_chunk_size: int
) -> None:
    """Save current file hashes and indexing configuration."""
    manifest_path = f"{DATA_PROCESSED_PATH}/manifest.json"
    data = {"max_chunk_size": max_chunk_size, "files": files}
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# Cette fonction récupère les anciens hashes.
def load_manifest() -> dict:
    manifest_path = f"{DATA_PROCESSED_PATH}/manifest.json"
    if not os.path.exists(manifest_path):
        return {
            "max_chunk_size": None,
            "files": {}
        }
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_changes(
    files: list[tuple[str, str]],
    old_manifest: dict,
    config: dict[str, str],
) -> tuple[list[tuple[str, str]], list[str], dict[str, str]]:
    new_manifest = {}
    changed_files = []
    old_files = old_manifest["files"]
    config_changed = old_manifest["config"] != config
    if config_changed:
        changed_files = files
    for file_path, content in files:
        file_hash = get_file_hash(content)
        new_manifest[file_path] = file_hash
        # in case file not exist in old_manifest(new_file) get return None
        if not config_changed and old_files.get(file_path) != file_hash:
            # Nouveaux fichiers et fichiers modifiés
            changed_files.append((file_path, content))
    deleted_files = [
        path for path in old_files
        if path not in new_manifest
    ]
    return changed_files, deleted_files, new_manifest



def load_old_data() -> dict[str,list[tuple]]:
    """Group old (chunk, text, embeddings) by file path."""
    if not (Path(DATA_PROCESSED_PATH) / "chunk.pkl").exists():
        return {}
    _, chunks, texts, embeddings = load_index()
    old_data: dict[str, list[tuple]] = {}
    for chunk, text, emb in zip(chunks, texts, embeddings):
        old_data.setdefault(chunk.file_path, []).append((chunk, text, emb))
    return old_data

def_build_index(max_chunk_size: int) -> None:
    files: list[tuple[str, str]] = load_files(DATA_RAW_PATH)
    changed_files, deleted_files, new_manifest = detecte_changes(
        files, load_manifest(), max_chunk_size
    )
    all_chunks: list[MinimalSource] = []
    all_texts: list[str] = []
    all_embeddings:list = []
    skip = {path for path, _ in changed_files} | deleted_files
    # restorer the old files qui ne sont pas changer
    for path, data in load_old_data().items():
        # modifié ou supprimé -> on ne recopie PAS ses anciens chunks
        if path in skip:
            continue
        for chunk, text, emp in data:
            all_chunks.append(chunk)
            all_texts.append(text)
            all_embeddings.append(emb)
    # rechunker les files qui sont modifier
    new_texts: list[]
    for path, content in tqdm(changed_files, desc="chunking_modified"):
        chunks = file_chunk(path, content, max_chunk_size)
        for chunk in chunks:
            text = content[chunk.first_character_index:chunk.last_character_index]
            all_chunks.append(chunk)
            all_texts.append(text)
            new_texts.append(text)
    if new_texts:
        all_embeddings.extend(Embedder().encode(new_texts))
    tokenized = [
        tokenize(text)
        for text in tqdm(all_texts, desc="Tokenizing")
    ]
    bm25 = BM25Okapi(tokenized)
    embeddings = (
        np.array(all_embeddings) if all_embeddings
        else np.empty((0, 384))
    )
    assert len(all_chunks) == len(all_texts) == len(embeddings), (
        "Mismatch between chunks, texts and embeddings!"
    )

    os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
    with open(f"{DATA_PROCESSED_PATH}/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(f"{DATA_PROCESSED_PATH}/chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(f"{DATA_PROCESSED_PATH}/texts.pkl", "wb") as f:
        pickle.dump(all_texts, f)
    with open(f"{DATA_PROCESSED_PATH}/embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    save_manifest(new_manifest, max_chunk_size)

    print(
        f"Ingestion complete! Indexed {len(all_chunks)} "
        f"chunks under {DATA_PROCESSED_PATH}/"
    )

def load_index() -> tuple[
    BM25Okapi,
    list[MinimalSource],
    list[str],
    np.ndarray
]:
    with open(f"{DATA_PROCESSED_PATH}/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    with open(f"{DATA_PROCESSED_PATH}/chunks.pkl", "rb") as f:
        chunks: list[MinimalSource] = pickle.load(f)

    with open(f"{DATA_PROCESSED_PATH}/texts.pkl", "rb") as f:
        texts: list[str] = pickle.load(f)

    with open(f"{DATA_PROCESSED_PATH}/embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    return bm25, chunks, texts, embeddings
