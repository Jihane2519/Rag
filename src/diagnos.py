import json
import sys
from collections import defaultdict
import pickle
from collections import defaultdict


def overlap(a_start, a_end, b_start, b_end):
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return union > 0 and inter / union >= 0.05


# def load_chunks_by_file(index_path):
#     """Construit {file_path: [(start, end), ...]} depuis ton index.

#     ADAPTE cette fonction à ton format d'index.
#     Ici on suppose un JSON : une liste de chunks avec
#     file_path, first_character_index, last_character_index.
#     """
#     with open(index_path) as f:
#         chunks = json.load(f)
#     by_file = defaultdict(list)
#     for c in chunks:
#         by_file[c["file_path"]].append(
#             (c["first_character_index"], c["last_character_index"])
#         )
#     return by_file


def load_chunks_by_file(index_path):
    with open(index_path, "rb") as f:
        chunks = pickle.load(f)

    by_file = defaultdict(list)

    for chunk in chunks:
        by_file[chunk.file_path].append(
            (
                chunk.first_character_index,
                chunk.last_character_index,
            )
        )

    return by_file



def diagnose(dataset_path, results_path, chunks_by_file, k=5):
    with open(dataset_path) as f:
        gt = json.load(f)["rag_questions"]
    with open(results_path) as f:
        res = {r["question_id"]: r
               for r in json.load(f)["search_results"]}

    no_chunk = ranked_low = found_count = 0
    for q in gt:
        retrieved = res[q["question_id"]]["retrieved_sources"][:k]
        for s in q["sources"]:
            found = any(
                s["file_path"] == r["file_path"]
                and overlap(s["first_character_index"],
                            s["last_character_index"],
                            r["first_character_index"],
                            r["last_character_index"])
                for r in retrieved
            )
            if found:
                found_count += 1
                continue
            covered = any(
                overlap(s["first_character_index"],
                        s["last_character_index"], a, b)
                for (a, b) in chunks_by_file.get(s["file_path"], [])
            )
            if covered:
                ranked_low += 1
            else:
                no_chunk += 1
                print("PAS DE CHUNK:", s["file_path"], "|",
                      q["question"][:80])

    print(f"trouvé: {found_count}")
    print(f"absent de l'index: {no_chunk}")
    print(f"présent mais mal classé: {ranked_low}")


if __name__ == "__main__":
    # usage: python -m src.diagnose INDEX DATASET RESULTS
    index_path, dataset_path, results_path = sys.argv[1:4]
    diagnose(dataset_path, results_path, load_chunks_by_file(index_path))
