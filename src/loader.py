from pathlib import Path
from tqdm import tqdm


def load_files(root: str) -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []

    all_files = [
        path
        for path in Path(root).rglob("*")
        if path.is_file() and path.suffix in {".md", ".py", ".txt"}
    ]

    for path in tqdm(all_files, desc="Files loading"):
        # Le contenu n'est pas valide en UTF-8.
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeEncodeError as e:
            print(f"Error: invalid UTF-8 encoding in {path}: {e}")
            continue
        files.append((str(path), content))
    return files
