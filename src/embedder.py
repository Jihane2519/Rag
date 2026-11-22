# from sentence_transformers import SentenceTransformer
# import numpy as np
# from src.config import MODEL_NAME_EMBEDDING


# class Embedder:
#     def __init__(self) -> None:
#         """Initialize the pre-trained embedding model on the CPU."""
#         # charger model deja entrainer
#         self.model = SentenceTransformer(
#             MODEL_NAME_EMBEDDING,
#             device="cpu"
#         )

#     def encode(self, texts: list[str]) -> np.ndarray:
#         """Convert a list of texts into normalized embedding vectors.

#         Args:
#             texts: List of input texts to encode.

#         Returns:
#             A NumPy array containing the embedding vectors.
#         """
#         # prendre texte en entree et retourner
#         # un vecteur embedding 384 dimensions
#         return self.model.encode(
#             texts,
#             convert_to_numpy=True,
#             normalize_embeddings=True,
#             show_progress_bar=False
#         )

from sentence_transformers import SentenceTransformer
import numpy as np
import os
import torch
from src.config import MODEL_NAME_EMBEDDING


class Embedder:
    def __init__(self) -> None:
        """Initialize the pre-trained embedding model on the CPU."""
        # charger model deja entrainer
        # utiliser tous les coeurs CPU disponibles
        torch.set_num_threads(os.cpu_count() or 1)
        self.model = SentenceTransformer(
            MODEL_NAME_EMBEDDING,
            device="cpu"
        )
        # limite native de all-MiniLM-L6-v2 (ne change pas les résultats)
        self.model.max_seq_length = 128

    def encode(self, texts: list[str]) -> np.ndarray:
        """Convert a list of texts into normalized embedding vectors.

        Args:
            texts: List of input texts to encode.

        Returns:
            A NumPy array containing the embedding vectors.
        """
        # prendre texte en entree et retourner
        # un vecteur embedding 384 dimensions
        return self.model.encode(
            texts,
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
