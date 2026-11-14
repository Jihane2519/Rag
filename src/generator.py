from typing import Any, cast

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BatchEncoding,
)

from src.models import MinimalSource
from src.config import MODEL_NAME


class Generator:
    """Gère le chargement et l'utilisation du modèle Qwen3."""

    def __init__(self) -> None:
        print(f"Loading model {MODEL_NAME}")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32
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
        context_parts: list[str] = []

        for chunk in chunks:
            for i, c in enumerate(all_chunks):
                if (
                    c.file_path == chunk.file_path
                    and c.first_character_index == chunk.first_character_index
                ):
                    context_parts.append(
                        f"Source: {c.file_path}\n{texts[i]}"
                    )
                    break

        context = "\n***\n".join(context_parts)

        prompt = f"""You are a helpful assistant.
        Answer the question using ONLY the information from the context.
        Be concise. Stop after one short paragraph.
        Context:
        {context}
        Question: {question}
        Answer:"""

        inputs: BatchEncoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2056
        )

        with torch.no_grad():
            output = cast(Any, self.model).generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.3
            )

        input_length = inputs["input_ids"].shape[1]

        answer = self.tokenizer.decode(
            output[0][input_length:],
            skip_special_tokens=True
        )

        if isinstance(answer, str):
            return answer.strip()

        if answer:
            return answer[0].strip()

        return ""
