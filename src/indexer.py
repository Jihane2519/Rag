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
from src.cache import clear_query_cache
from typing import Any, cast
_index_cache = None


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


def save_manifest(files: dict[str, str], max_chunk_size: int) -> None:
    """Save current file hashes and indexing configuration."""
    manifest_path = f"{DATA_PROCESSED_PATH}/manifest.json"
    data = {"max_chunk_size": max_chunk_size, "files": files}
    with open(manifest_path, "w", encoding="utf-8") as f:
        # Sérialisation dans un fichier (Disque)
        json.dump(data, f, indent=4)


# Cette fonction récupère les anciens hashes.
def load_manifest() -> dict[str, Any]:
    manifest_path = f"{DATA_PROCESSED_PATH}/manifest.json"
    if not os.path.exists(manifest_path):
        return {
            "max_chunk_size": None,
            "files": {}
        }
    with open(manifest_path, "r", encoding="utf-8") as f:
        return cast(dict[str, Any], json.load(f))


def detect_changes(
    files: list[tuple[str, str]],
    old_manifest: dict,
    max_chunk_size: int,
) -> tuple[list[tuple[str, str]], list[str], dict[str, str]]:
    new_manifest = {}
    changed_files = []
    old_files = old_manifest["files"]
    config_changed = old_manifest["max_chunk_size"] != max_chunk_size
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


def load_old_data() -> dict[str, list[tuple]]:
    """Group old (chunk, text, embeddings) by file path."""
    if not (Path(DATA_PROCESSED_PATH) / "chunks.pkl").exists():
        return {}
    _, chunks, texts, embeddings = load_index()
    old_data: dict[str, list[tuple]] = {}
    for chunk, text, emb in zip(chunks, texts, embeddings):
        old_data.setdefault(chunk.file_path, []).append((chunk, text, emb))
    return old_data


def build_index(max_chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
    global _index_cache
    files: list[tuple[str, str]] = load_files(DATA_RAW_PATH)
    # files: list[tuple[str, str]] = load_files("data/test_empty")
    changed_files, deleted_files, new_manifest = detect_changes(
        files, load_manifest(), max_chunk_size
    )
    all_chunks: list[MinimalSource] = []
    all_texts: list[str] = []
    all_embeddings: list = []
    skip = {path for path, _ in changed_files} | set(deleted_files)
    # restorer the old files qui ne sont pas changer
    for path, data in load_old_data().items():
        # modifié ou supprimé -> on ne recopie PAS ses anciens chunks
        if path in skip:
            continue
        for chunk, text, emb in data:
            all_chunks.append(chunk)
            all_texts.append(text)
            all_embeddings.append(emb)
    # rechunker les files qui sont modifier
    new_texts: list = []
    for path, content in tqdm(changed_files, desc="chunking_modified"):
        chunks = chunk_file(path, content, max_chunk_size)
        for chunk in chunks:
            text = content[
                chunk.first_character_index:chunk.last_character_index
            ]
            all_chunks.append(chunk)
            all_texts.append(text)
            new_texts.append(text)
    if new_texts:
        all_embeddings.extend(Embedder().encode(new_texts))
    # if not all_texts:
    #     save_manifest(new_manifest, max_chunk_size)
    #     _index_cache = None
    #     print("No files to index. Keeping the existing index.")
    #     return
    if not all_texts:
        save_manifest(new_manifest, max_chunk_size)
        _index_cache = None
        clear_query_cache()

        # Supprimer les anciens fichiers de l'index
        for filename in (
            "bm25.pkl",
            "chunks.pkl",
            "texts.pkl",
            "embeddings.pkl"
        ):
            file_path = Path(DATA_PROCESSED_PATH) / filename
            # supprime le fichier du disque
            file_path.unlink(missing_ok=True)

        print("Index is empty. Old index files removed.")
        return
    tokenized = [
        tokenize(text)
        for text in tqdm(all_texts, desc="Tokenizing")
    ]
    embeddings = (
        np.array(all_embeddings) if all_embeddings
        else np.empty((0, 384))
    )

    assert len(all_chunks) == len(all_texts) == len(embeddings), (
        "Mismatch between chunks, texts and embeddings!"
    )

    bm25 = BM25Okapi(tokenized)

    os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
    #  Écriture dans un fichier Disque
    with open(f"{DATA_PROCESSED_PATH}/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(f"{DATA_PROCESSED_PATH}/chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(f"{DATA_PROCESSED_PATH}/texts.pkl", "wb") as f:
        pickle.dump(all_texts, f)
    with open(f"{DATA_PROCESSED_PATH}/embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    save_manifest(new_manifest, max_chunk_size)
    _index_cache = None
    clear_query_cache()
    # print("Chunks:", len(all_chunks))
    # print("Texts:", len(all_texts))
    # print("Embeddings:", len(all_embeddings))
    # print(
    #     f"Ingestion complete! Indexed {len(all_chunks)} "
    #     f"chunks under {DATA_PROCESSED_PATH}/"
    # )


def load_index() -> tuple[
    BM25Okapi,
    list[MinimalSource],
    list[str],
    np.ndarray
]:
    global _index_cache
    # Vérifier si l'index existe déjà dans le cache
    if _index_cache is not None:
        return _index_cache
    # print("Loading index from disk...")
    required_files = (
        "bm25.pkl",
        "chunks.pkl",
        "texts.pkl",
        "embeddings.pkl"
    )

    for filename in required_files:
        path = Path(DATA_PROCESSED_PATH) / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{filename} not found. Run build_index() first."
            )
    # Sinon, charger les fichiers depuis le disque
    with open(f"{DATA_PROCESSED_PATH}/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
        # load() chargement de l'objet en memoire RAM
        # RAM(on l'utilise comme cache)

    with open(f"{DATA_PROCESSED_PATH}/chunks.pkl", "rb") as f:
        chunks: list[MinimalSource] = pickle.load(f)
    # Lecture depuis un fichier (Disque)
    with open(f"{DATA_PROCESSED_PATH}/texts.pkl", "rb") as f:
        texts: list[str] = pickle.load(f)

    with open(f"{DATA_PROCESSED_PATH}/embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    # Sauvegarder l'index charge dans le cache RAM
    _index_cache = (bm25, chunks, texts, embeddings)
    return _index_cache
