from src.models import MinimalSource
import ast
import re
from src.models import MinimalSource

from src.models import MinimalSource

#73  & 45
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

# def _flush_chunk(file_path, content, start, text, max_size, chunks):
#     if len(text) <= max_size:
#         chunks.append(
#                 MinimalSource(
#                     file_path=file_path,
#                     first_character_index=start,
#                     last_character_index=start + len(text)
#                 )
#         )
#     else:
#         offset = 0
#         while offset < len(text):
#             end = min(offset + max_size, len(text))
#             chunks.append(
#                 MinimalSource(
#                     file_path=file_path,
#                     first_character_index=start + offset,
#                     last_character_index=start + end
#                 )
#             )
#             offset = end


# def chunk_markdown(file_path, content, max_size) -> list[MinimalSource]:
#     # max_size = int(max_size)
#     chunks = []
#     lines = content.split("\n")
#     current_start = 0
#     current_text = ""
#     for line in lines:
#         if line.startswith("#") and current_text.strip():
#             _flush_chunk(
#                 file_path,
#                 content,
#                 current_start,
#                 current_text,
#                 max_size,
#                 chunks
#             )
#             current_start = current_start + len(current_text)
#             current_text = line + "\n"
#         else:
#             current_text += line + "\n"
#     if current_text.strip():
#         _flush_chunk(
#             file_path,
#             content,
#             current_start,
#             current_text,
#             max_size,
#             chunks
#         )
#     return chunks


def _flush_chunk(file_path, start, end, max_size, chunks):
    """Découpe une section en chunks avec des positions exactes."""
    offset = start

    while offset < end:
        chunk_end = min(offset + max_size, end)

        chunks.append(
            MinimalSource(
                file_path=file_path,
                first_character_index=offset,
                last_character_index=chunk_end,
            )
        )

        offset = chunk_end


def chunk_markdown(
    file_path: str,
    content: str,
    max_size: int = 2000,
) -> list[MinimalSource]:
    """Découpe un fichier Markdown en conservant les positions originales."""
    max_size = int(max_size)

    if max_size <= 0:
        raise ValueError("max_size doit être supérieur à 0")

    chunks: list[MinimalSource] = []
    section_start = 0
    position = 0

    for line in content.splitlines(keepends=True):
        if line.lstrip().startswith("#") and position > section_start:
            _flush_chunk(
                file_path,
                section_start,
                position,
                max_size,
                chunks,
            )
            section_start = position

        position += len(line)

    if section_start < len(content):
        _flush_chunk(
            file_path,
            section_start,
            len(content),
            max_size,
            chunks,
        )

    return chunks


def chunk_file(file_path, content, max_size) -> list[MinimalSource]:
    if file_path.endswith(".py"):
        return chunk_python(file_path, content, max_size)
    elif file_path.endswith(".md"):
        return chunk_markdown(file_path, content, max_size)
    elif file_path.endswith("CMakeLists.txt"):
        return chunk_markdown(file_path, content, max_size)
    else:
        return []
