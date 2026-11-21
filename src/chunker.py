
from langchain_text_splitters import (
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    TextSplitter,
)

from src.config import DEFAULT_CHUNK_SIZE
from src.config import DEFAULT_OVERLAP
from src.models import MinimalSource


def chunk_file(
    file_path: str,
    content: str,
    max_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[MinimalSource]:
    """Split a file into chunks and return their source positions.

    Selects a splitter based on the file extension and verifies
    that each chunk matches its corresponding content in the original file.

    Args:
        file_path: Path of the file to split.
        content: Original content of the file.
        max_size: Maximum size of each chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        A list of MinimalSource objects containing file paths
        and character positions.
    """
    if max_size <= 0:
        raise ValueError("max_size doit être supérieur à 0")

    if overlap < 0 or overlap >= max_size:
        raise ValueError("overlap doit être compris entre 0 et max_size - 1")

    splitter: TextSplitter

    if file_path.endswith((".md", ".txt")):
        splitter = MarkdownTextSplitter(
            chunk_size=max_size,
            chunk_overlap=overlap,
            add_start_index=True,
        )
    elif file_path.endswith(".py"):
        splitter = PythonCodeTextSplitter(
            chunk_size=max_size,
            chunk_overlap=overlap,
            add_start_index=True,
        )
    else:
        return []

    documents = splitter.create_documents([content])
    chunks: list[MinimalSource] = []

    for document in documents:
        text = document.page_content
        start = document.metadata["start_index"]
        end = start + len(text)

        # Vérifier que les indices correspondent au contenu original.
        if content[start:end] != text:
            raise ValueError(
                f"Indices incorrects pour {file_path}: {start}:{end}"
            )

        chunks.append(
            MinimalSource(
                file_path=file_path,
                first_character_index=start,
                last_character_index=end
            )
        )

    return chunks
