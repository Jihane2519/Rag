from src.models import MinimalSource


def chunk_python(file_path, content, max_size) -> list[MinimalSource]:
    chunks = []
    start = 0
    while start < len(content):
        end = min(start + max_size, len(content))
        chunks.append(MinimalSource(file_path=file_path, first_character_index=start, last_character_index=end))
        start = end
    return chunks


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
