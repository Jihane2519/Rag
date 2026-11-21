from pathlib import Path
from tqdm import tqdm


def load_files(root: str) -> list[tuple[str, str]]:
    """Load supported files from a directory and its subdirectories.

    Only Markdown, Python, and text files are loaded.

    Args:
        root: Path to the directory containing the files.

    Returns:
        A list of tuples containing each file path and its content.
    """
    files: list[tuple[str, str]] = []
    root_path = Path(root)

    if not root_path.exists():
        print(f"Error: directory does not exist: {root}")
        return files

    if not root_path.is_dir():
        print(f"Error: path is not a directory: {root}")
        return files

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
