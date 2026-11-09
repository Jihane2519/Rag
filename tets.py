import hashlib


def get_file_hash(content: str) -> str:
    # get the 64 character hexadicimale string
    # ex output dffd6021bb2bd5b0af6762d8822ee3a61045cdb3e05a5018e00d55a1873020e3
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def detect_changes(
    files: list[tuple[str, str]],
    old_manifest: dict,
) -> tuple[list[tuple[str, str]], list[str], dict[str, str]]:
    new_manifest = {}
    changed_files: list[tuple[str, str]] = []
    deleted_files = []
    for file_path, content in files:
        file_hash = get_file_hash(content)
        new_manifest[file_path] = file_hash
        if old_manifest.get(file_path) != file_hash:
            changed_files.append((file_path, content))
    deleted_files = [
        file for file in old_manifest
        if file not in new_manifest
    ]
    return changed_files, deleted_files, new_manifest

def load_manifest() -> dict:
    manifest_path = f"{DATA_PROCESSED_PATH}/manifest.json"
    if not os.path.exists(manifest_path):
        return {
            "max_chunk_size": None,
            "files": {}
        }
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(files: dict[str, str], max_chunk_size: int) -> None:
    data = {"max_chunk_size": max_chunk_size, "files": files}
    MANIFEST_PATH = f"{DATA_PROCESSED_PATH}/manifest.json"
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def load_old_data() -> dict[str,list[tuple]]:
    """Group old (chunk, text, embeddings) by file path."""
    if not (Path(DATA_PROCESSED_PATH) / "chunk.pkl").exists():
        return {}
    _, chunks, texts, embedding = load_index()
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
        


# def build_index(max_chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
#     files = load_files(DATA_RAW_PATH)
#     all_chunks: list[MinimalSource] = []
#     # pour le texte entre start et end
#     all_texts: list[str] = []
#     # le text chnger pour le donner a encode
#     new_texts: list[str] = []
#     all_embeddings = []
#     old_data = {}
#     old_manifest = load_manifest()
#     current_config = {
#         "max_chunk_size": max_chunk_size,
#         "overlap": 200
#     }
#     config_changed = old_manifest["config"] != current_config
#     changed_files, deleted_files, new_manifest = detect_changes(
#         files,
#         old_manifest,
#     )
#     if config_changed:
#         print("Chunker configuration changed. Re-indexing all files ...")
#         changed_files = files
#     if (Path(DATA_PROCESSED_PATH) / "chunks.pkl").exists():
#         _, old_chunks, old_texts, old_embeddings = load_index()
#     else:
#         old_chunks = []
#         old_texts = []
#         old_embeddings = np.empty((0, 384))

#     for chunk, text, embedding in zip(
#         old_chunks,
#         old_texts,
#         old_embeddings
#     ):
#         old_data.setdefault(chunk.file_path, []).append(
#             (chunk, text, embedding)
#         )
#     # solution car on ne peut pas dire file path int (file_path, content)
#     # ou bien on va returner seulement les files si ona pas utiliser les contents
#     changed_paths = {file_path for file_path, content in changed_files}
#     for file_path, data in old_data.items():
#         if file_path not in changed_paths and file_path not in deleted_files:
#             for chunk, text, embedding in data:
#                 all_chunks.append(chunk)
#                 all_texts.append(text)
#                 all_embeddings.append(embedding)
#     for file_path, content in tqdm(changed_files, desc="Chunking"):
#         chunks = chunk_file(file_path, content, max_chunk_size)
#         for chunk in chunks:
#             all_chunks.append(chunk)
#             # extraire le texte du chunk
#             chunk_text = content[
#                 chunk.first_character_index:chunk.last_character_index
#             ]
#             all_texts.append(chunk_text)
#             new_texts.append(chunk_text)
#     print("Building BM25 index...")
#     # ex: [["python", "is", "powerful"]]
#     # Chaque liste représente un document/chunk.
#     # tokenized_docs = [text.lower().split() for text in all_texts]
#     # tokenized_docs = [
#     #     re.findall(r'\w+', text.lower())
#     #     for text in tqdm(all_texts, desc="Tokenizing")
#     # ]
#     tokenized = [
#         tokenize(text)
#         for text in tqdm(all_texts, desc="Tokenizing")
#     ]
#     bm25 = BM25Okapi(tokenized)
#     if new_texts:
#         embedder = Embedder()
#         new_embeddings = embedder.encode(new_texts)
#         all_embeddings.extend(new_embeddings)
#     # car all_embeddings est une list []
#     embeddings = np.array(all_embeddings)
#     # bm25_docs = BM25Okapi(tokenized_docs)
#     # bm25_code = BM25Okapi(tokenized_code)
#     # print(type(bm25)) -> <class 'rank_bm25.BM25Okapi'>

#     os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)

#     # with open(f"{DATA_PROCESSED_PATH}/bm25_docs.pkl", "wb") as f:
#     #     pickle.dump(bm25_docs, f)
#     # with open(f"{DATA_PROCESSED_PATH}/bm25_code.pkl", "wb") as f:
#     #     pickle.dump(bm25_code, f)
#     with open(f"{DATA_PROCESSED_PATH}/bm25.pkl", "wb") as f:
#         pickle.dump(bm25, f)
#     with open(f"{DATA_PROCESSED_PATH}/chunks.pkl", "wb") as f:
#         pickle.dump(all_chunks, f)
#     with open(f"{DATA_PROCESSED_PATH}/texts.pkl", "wb") as f:
#         pickle.dump(all_texts, f)
#     with open(f"{DATA_PROCESSED_PATH}/embeddings.pkl", "wb") as f:
#         pickle.dump(embeddings, f)
#     save_manifest(new_manifest, current_config)

#     print(
#         f"Ingestion complete! Indexed {len(all_chunks)} "
#         f"chunks under {DATA_PROCESSED_PATH}/"
#     )
#     print("Chunks:", len(all_chunks))
#     print("Texts:", len(all_texts))
#     print("Embeddings:", len(all_embeddings))
#     assert len(all_chunks) == len(all_texts) == len(all_embeddings), (
#         "Mismatch between chunks, texts and embeddings!"
#     )
