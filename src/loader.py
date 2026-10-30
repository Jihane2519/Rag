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
        content = path.read_text(encoding="utf-8")
        files.append((str(path), content))
    return files

# import os
# def load_files(raw_path: str) -> list[tuple[str, str]]:
#     files = []
#     all_files = []
#     for root, _, filenames in os.walk(raw_path):
#         for filename in filenames:
#             if filename.endswith((".py", ".md", "CMakeLists.txt")):
#                 all_files.append(os.path.join(root, filename))
#     for file_path in tqdm(all_files, desc="files is loading"):
#         # print(file_path)
#         try:
#             with open(file_path, "r", encoding="utf-8") as f:
#                 content = f.read()
#             files.append((file_path, content))
#         except Exception as e:
#             print(e)
#     return files
#     # retourne [(file_path, content), ...]
#     # file_path = "data/raw/vllm-0.10.1/..."
