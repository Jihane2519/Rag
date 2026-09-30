# src/generator.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.models import MinimalSource

MODEL_NAME = "Qwen/Qwen3-0.6B"


class Generator:
    """Gère le chargement et l'utilisation du modèle Qwen3."""

    def __init__(self) -> None:
        """Initialise le générateur et charge le modèle."""
        print(f"Loading model {MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32  # CPU friendly
        )
        self.model.eval()
        print("Model loaded!")

    def generate_answer(
        self,
        question: str,
        chunks: list[MinimalSource],
        texts: list[str],
        all_chunks: list[MinimalSource]
    ) -> str:
        """
        Génère une réponse à partir de la question et des chunks récupérés.

        Args:
            question: la question posée
            chunks: les MinimalSource trouvés par le retriever
            texts: tous les textes indexés
            all_chunks: tous les chunks indexés

        Returns:
            la réponse générée par Qwen3
        """
        # 1. construire le contexte à partir des chunks trouvés
        context_parts = []
        for chunk in chunks:
            for i, c in enumerate(all_chunks):
                if (c.file_path == chunk.file_path and
                        c.first_character_index == chunk.first_character_index):
                    context_parts.append(
                        f"Source: {c.file_path}\n{texts[i]}"
                    )
                    break

        context = "\n\n---\n\n".join(context_parts)

        # 2. construire le prompt
        prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:"""

        # 3. tokenizer le prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        # 4. générer la réponse
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,  # réponse déterministe
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 5. décoder (seulement les nouveaux tokens)
        input_length = inputs["input_ids"].shape[1]
        answer = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )

        return answer.strip()