from src.models import MinimalSource
import ast


# 73  & 45
def chunk_python(file_path, content, max_size=2000):
    max_size = int(max_size)
    chunks = []
    try:
        tree = ast.parse(content)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                # calculer start/end en caractères
                lines = content.split("\n")
                start_char = sum(len(l)+1 for l in lines[:node.lineno-1])
                end_char = sum(len(l)+1 for l in lines[:node.end_lineno])
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
