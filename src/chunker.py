from src.models import MinimalSource
import ast
import re
from src.models import MinimalSource


def chunk_python(file_path, content, max_size) -> list[MinimalSource]:
    chunks = []
    start = 0
    while start < len(content):
        end = min(start + 2000, len(content))
        chunks.append(MinimalSource(file_path=file_path, first_character_index=start, last_character_index=end))
        start = end
    return chunks


def chunk_python(file_path, content, max_size=2000):
    # max_size = int(max_size)
    chunks = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                start = node.col_offset
                # calculer start/end en caractères
                lines = content.split("\n")
                start_char = sum(len(l)+1 for l in lines[:node.lineno-1])
                end_char = sum(len(l)+1 for l in lines[:node.end_lineno])
                # if end_char - start_char <= max_size:
                #     chunks.append(MinimalSource(
                #         file_path=file_path,
                #         first_character_index=start_char,
                #         last_character_index=end_char
                #     ))
                offset = start_char

                while offset < end_char:
                    chunk_end = min(offset + max_size, end_char)

                    chunks.append(
                        MinimalSource(
                            file_path=file_path,
                            first_character_index=offset,
                            last_character_index=chunk_end
                        )
                    )

                    offset = chunk_end
    except Exception:
        # fallback → split par taille fixe
        start = 0
        while start < len(content):
            end = min(start + max_size, len(content))
            chunks.append(MinimalSource(
                file_path=file_path,
                first_character_index=start,
                last_character_index=end
            ))
            start = end
    return chunks


def chunk_python(
    file_path: str,
    content: str,
    max_size: int = 2000
) -> list[MinimalSource]:
    """Découpe un fichier Python par fonctions/classes avec ast."""
    max_size = int(max_size)
    chunks = []
    lines = content.split("\n")

    try:
        tree = ast.parse(content)
        nodes = [
            node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]

        for node in nodes:
            # calculer start en caractères
            start_char = sum(len(l) + 1 for l in lines[:node.lineno - 1])
            end_char = sum(len(l) + 1 for l in lines[:node.end_lineno])

            # si trop grand → split par taille
            if end_char - start_char > max_size:
                pos = start_char
                while pos < end_char:
                    end = min(pos + max_size, end_char)
                    chunks.append(MinimalSource(
                        file_path=file_path,
                        first_character_index=pos,
                        last_character_index=end
                    ))
                    pos = end - 200  # overlap
                    if pos >= end_char:
                        break
            else:
                chunks.append(MinimalSource(
                    file_path=file_path,
                    first_character_index=start_char,
                    last_character_index=end_char
                ))

    except Exception:
        # fallback → split par taille fixe avec overlap
        start = 0
        while start < len(content):
            end = min(start + max_size, len(content))
            chunks.append(MinimalSource(
                file_path=file_path,
                first_character_index=start,
                last_character_index=end
            ))
            if end == len(content):
                break
            start = end - 200

    # dédupliquer les chunks
    seen = set()
    unique_chunks = []
    for c in chunks:
        key = (c.first_character_index, c.last_character_index)
        if key not in seen:
            seen.add(key)
            unique_chunks.append(c)

    return unique_chunks

def _flush_chunk(file_path, content, start, text, max_size, chunks):
    if len(text) <= max_size:
        chunks.append(
                MinimalSource(
                    file_path=file_path,
                    first_character_index=start,
                    last_character_index=start + len(text)
                )
        )
    else:
        offset = 0
        while offset < len(text):
            end = min(offset + max_size, len(text))
            chunks.append(
                MinimalSource(
                    file_path=file_path,
                    first_character_index=start + offset,
                    last_character_index=start + end
                )
            )
            offset = end


def chunk_markdown(file_path, content, max_size) -> list[MinimalSource]:
    # max_size = int(max_size)
    chunks = []
    lines = content.split("\n")
    current_start = 0
    current_text = ""
    for line in lines:
        if line.startswith("#") and current_text.strip():
            _flush_chunk(
                file_path,
                content,
                current_start,
                current_text,
                max_size,
                chunks
            )
            current_start = current_start + len(current_text)
            current_text = line + "\n"
        else:
            current_text += line + "\n"
    if current_text.strip():
        _flush_chunk(
            file_path,
            content,
            current_start,
            current_text,
            max_size,
            chunks
        )
    return chunks


def chunk_file(file_path, content, max_size) -> list[MinimalSource]:
    if file_path.endswith(".py"):
        return chunk_python(file_path, content, max_size)
    elif file_path.endswith(".md"):
        return chunk_markdown(file_path, content, max_size)
    else:
        return []
