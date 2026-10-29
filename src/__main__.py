import json
import os
from tqdm import tqdm
import fire
from src.retriever import search
from src.indexer import build_index, load_index
from src.generator import Generator
from src.models import (
    MinimalSearchResults,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    RagDataset,
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
        save_directory: str = "datr q in tqdm(dataa/output/search_results"
    ) -> None:
        try:
            with open(dataset_path, "r") as f:
                data = json.load(f)
            dataset = RagDataset(**data)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return
        all_results: list[MinimalSearchResults]= []
        tokenizer = (
            "code"
            if "code" in os.path.basename(dataset_path).lower()
            else "docs"
        )
        for q in tqdm(dataset.rag_questions, desc="Searching questions"):
            sources = search(q.question, k, tokenizer=tokenizer)
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
        bm25, all_chunks, texts = load_index()
        generator = Generator()
        # 3. générer les réponses
        all_answers = []
        for result in tqdm(search_results.search_results, desc="Generating answers"):
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
        mesurer la qualité de ton Retriever, en calculant le Recall@k

        student_search_results_path: str
        C'est le chemin du fichier contenant les résultats obtenus par ton système.

        dataset_path: str
        C'est le chemin vers le dataset de référence,
        qui contient les questions et leurs sources correctes.

        """
        # 1.charger tes resultats
        try:
            with open(student_search_results_path, "r") as f:
                student_data = json.load(f)
                student_results = StudentSearchResults(**student_data)
        except Exception as e:
            print(f"Error loading student results: {e}")
            return
        # 2. charger le dataset de reference
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
                    # Si au moins 5% du contenu est considéré comme commun, on considère 
                    # que la source trouvée correspond à la source de référence.
                    if IoU > 0.05:
                        found += 1
                        # Une fois qu'on a trouvé une source 
                        # de référence qui correspond à une source retrouvée
                        break
            recall = found / len(ref_q.sources) if ref_q.sources else 0
            total_recall += recall
            count += 1
        avg_recall = total_recall / count if count > 0 else 0
        print(f"Recall@{k}: {avg_recall:.3f} ({avg_recall * 100:.1f}%)")


if __name__ == "__main__":
    fire.Fire(RAG)
