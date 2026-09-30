import json
import os
from tqdm import tqdm
import fire

from src.chunker import chunk_file
from src.retriever import search
from src.indexer import build_index, load_index
from src.generator import Generator
from src.models import (
    MinimalSource,
    MinimalSearchResults,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    RagDataset,
    UnansweredQuestion,
    AnsweredQuestion,
)

class RAG:
    """CLI principal du système RAG."""
    def index(self, max_chunk_size: int = 2000) -> None:
        """
        Ingeste data/raw/ et construit l'index sous data/processed/.

        Args:
            max_chunk_size: taille max d'un chunk en caractères
        """
        build_index(max_chunk_size)
    def search(self, query: str, k: int = 5) -> None:
        if not query:
            print("Error: query is empty")
            return
        if k <= 0:
            print("Error: k must be > 0")
            return
        results = search(query, k)
        for r in results:
            print(
                f"{r.file_path} [{r.first_character_index}:{r.last_character_index}]"
            )
    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = "data/output/search_results"
    ) -> None:
        try:
            with open(dataset_path, "r") as f:
                data = json.load(f)
            dataset = RagDataset(**data)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return
        all_results: list[MinimalSearchResults]= []
        for q in tqdm(dataset.rag_questions, desc="serching_questions"):
            sources = search(q.question, k)
            all_results.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=sources
                )
            )
            output = StudentSarchResults(
                search_results=all_results,
                k=k
            )
            os.makedirs(save_directory, exist_ok=True)
            # le nom de fichier seulement 
            filename = os.path.basename(dataset_path)
            save_path = os.path.join(save_directory, filename)
            with open(save_path, "w") as f:
                f.write(output.model_dump_json(indent=2))
            print(f"Saved student_search_results to {save_path}")


    def answer(self, query: str, k: int = 5) -> None:
        if not query:
            print("Error: query is empty")
            return
        if k <= 0:
            print("Error: k must be > 0")
            return
        bm25, all_chunks, texts = load_index()
        chunks = search(query, k)
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
        # with open(student_search_results_path, "r") as f:
        #     data = json.load(f)
        #     search_results = StudentSearchResults(**data)
        #     bm25, all_chunks, texts = load_index()
        #     generator = Generator()
        # for result in tqdm(
        #     search_results.search_results,
        #     desc="Generating answers"
        # ):
    def evaluate(self, student_search_results_path, dataset_path)


if __name__== "__main__":
    # fire.Fire(RAG)
    with open("/goinfre/jbounou/rag/tester/test.md", "r") as e:
        res = e.read()
    chunks = chunk_file("/goinfre/jbounou/rag/test", res, 100)
    build_index(1000)
    meilleur = search("What is BM25 used for?", 1)
    start = meilleur[0].first_character_index
    end = meilleur[0].last_character_index
    print(res[start:end])
    # print(res)
# def main() -> None:
#     print("RAG against the machine")
# if __name__ == "__main__":
#     main()