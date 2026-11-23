import json
import os
from tqdm import tqdm
# from typing import TYPE_CHECKING

# if TYPE_CHECKING:
import fire
# else:
#     import fire

from src.retriever import search_re
from src.indexer import build_index, load_index
from src.generator import Generator
from src.embedder import Embedder
from src.config import DEFAULT_CHUNK_SIZE
from src.models import (
    MinimalSearchResults,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    RagDataset,
    AnsweredQuestion,
)


class RAG:
    """Main CLI for the RAG system."""

    def index(self, max_chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        """
        Ingeste data/raw/ et construit l'index sous data/processed/.

        Args:
            max_chunk_size: taille max d'un chunk en caractères
        """
        build_index(max_chunk_size)
        # clear_query_cache()

    def search(self, query: str, k: int = 5) -> None:
        """Search for the most relevant chunks for a given query.

        Args:
            query: The user's search query.
            k: Maximum number of results to retrieve.
        """
        if not query:
            print("Error: query is empty")
            return
        if not isinstance(k, int) or k <= 0:
            print("Error: k must be a positive integer")
            return
        if not query.strip():
            print("query must not be empty")
            return
        results = search_re(query, k)

        for r in results:
            print(
                f"{r.file_path}"
                f"[{r.first_character_index}:{r.last_character_index}]"
            )

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = "data/output/search_results"
    ) -> None:
        """Retrieve relevant sources for each question in a dataset.

        Save the search results as a JSON file.

        Args:
            dataset_path: Path to the input dataset JSON file.
            k: Maximum number of sources to retrieve per question.
            save_directory: Directory where the results are saved.
        """
        try:
            with open(dataset_path, "r") as f:
                data = json.load(f)
            dataset = RagDataset(**data)
            if not dataset.rag_questions:
                raise ValueError("Error: dataset is empty")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

        if not isinstance(k, int) or k <= 0:
            print("Error: k must be a positive integer")
            return

        all_results: list[MinimalSearchResults] = []

        # tokenizer = (
        #     "code"
        #     if "code" in os.path.basename(dataset_path).lower()
        #     else "docs"
        # )

        index = load_index()
        embedder = Embedder()

        for q in tqdm(dataset.rag_questions, desc="Searching questions"):
            if not q.question.strip():
                print(f"Skipping empty question: {q.question_id}")
                continue
            sources = search_re(q.question, k, embedder, index)

            all_results.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=sources
                )
            )

        output = StudentSearchResults(
            search_results=all_results,
            k=k
        )

        os.makedirs(save_directory, exist_ok=True)

        filename = os.path.basename(dataset_path)
        save_path = os.path.join(save_directory, filename)

        with open(save_path, "w") as f:
            f.write(output.model_dump_json(indent=2))

        print(f"Saved student_search_results to {save_path}")

    def answer(self, query: str, k: int = 5) -> None:
        """Generate an answer using the most relevant retrieved chunks.

        Args:
            query: The user's question.
            k: Maximum number of sources to retrieve.
        """
        if not query.strip():
            print("Error: query is empty")
            return

        if not isinstance(k, int) or k <= 0:
            print("Error: k must be a positive integer")
            return

        _, all_chunks, texts, _ = load_index()

        chunks = search_re(query, k)
        generator = Generator()

        reponse_finale = generator.generate_answer(
            query,
            chunks,
            texts,
            all_chunks
        )

        print(f"Answer : {reponse_finale}")

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer"
    ) -> None:
        """
        Génère les réponses pour un dataset et sauvegarde les résultats.

        Args:
            student_search_results_path: chemin vers StudentSearchResults JSON
            save_directory: dossier de sauvegarde
        """
        try:
            with open(student_search_results_path, "r") as f:
                data = json.load(f)
            search_results = StudentSearchResults(**data)
        except Exception as e:
            print(f"Error loading search results: {e}")
            return

        # 2. charger index et générateur UNE seule fois
        _, all_chunks, texts, _ = load_index()
        generator = Generator()

        # 3. générer les réponses
        all_answers = []

        for result in tqdm(
            search_results.search_results, desc="Generating answers"
        ):
            if not result.question.strip():
                print(f"Skipping empty question: {result.question_id}")
                continue
            ans = generator.generate_answer(
                result.question,
                result.retrieved_sources,
                texts,
                all_chunks
            )

            all_answers.append(MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=ans
            ))

        # 4. sauvegarder
        output = StudentSearchResultsAndAnswer(
            search_results=all_answers,
            k=search_results.k
        )

        os.makedirs(save_directory, exist_ok=True)

        filename = os.path.basename(student_search_results_path)
        save_path = os.path.join(save_directory, filename)

        with open(save_path, "w") as f:
            f.write(output.model_dump_json(indent=2))

        print(f"Saved student_search_results_and_answer to {save_path}")

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str
    ) -> None:
        """
        Mesure la qualité du Retriever en calculant le Recall@k.
        """

        # 1. charger les résultats
        try:
            with open(student_search_results_path, "r") as f:
                student_data = json.load(f)
                student_results = StudentSearchResults(**student_data)
        except Exception as e:
            print(f"Error loading student results: {e}")
            return

        # 2. charger le dataset de référence
        try:
            with open(dataset_path, "r") as f:
                ref_data = json.load(f)
            ref_dataset = RagDataset(**ref_data)
        except Exception as e:
            print(f"Error loading reference dataset: {e}")
            return

        # 3. construire un dict question_id -> vraies sources
        ref_dict: dict[str, AnsweredQuestion] = {}

        for q in ref_dataset.rag_questions:
            if isinstance(q, AnsweredQuestion):
                ref_dict[q.question_id] = q

        k = student_results.k
        total_recall = 0.0
        count = 0

        for result in student_results.search_results:
            if result.question_id not in ref_dict:
                continue

            ref_q = ref_dict[result.question_id]
            found = 0

            for ref_source in ref_q.sources:
                for student_source in result.retrieved_sources:
                    if student_source.file_path != ref_source.file_path:
                        continue

                    inter_start = max(
                        student_source.first_character_index,
                        ref_source.first_character_index
                    )

                    inter_end = min(
                        student_source.last_character_index,
                        ref_source.last_character_index
                    )

                    inter = max(0, inter_end - inter_start)

                    union = (
                        student_source.last_character_index
                        - student_source.first_character_index
                        + ref_source.last_character_index
                        - ref_source.first_character_index
                        - inter
                    )

                    # IoU (Intersection over Union)
                    IoU = inter / union if union > 0 else 0

                    # Si au moins 5% du contenu est considéré
                    # comme commun, on considère que la source
                    # trouvée correspond à la source de référence.
                    if IoU > 0.05:
                        found += 1
                        break

            recall = found / len(ref_q.sources) if ref_q.sources else 0
            total_recall += recall
            count += 1

        avg_recall = total_recall / count if count > 0 else 0

        print(f"Recall@{k}: {avg_recall:.3f} ({avg_recall * 100:.1f}%)")


if __name__ == "__main__":
    try:
        fire.Fire(RAG)
    except KeyboardInterrupt:
        print("\nOperation interrupted by user.")
        raise SystemExit(130)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise SystemExit(1)
    except (OSError, ValueError) as e:
        print(f"Error: {e}")
        raise SystemExit(1)
