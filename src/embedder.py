from sentence_transformers import SentenceTransformer
from src.config import MODEL_NAME_EMBEDDING


class Embedder:
    def __init__(self) -> None:
        # charger model deja entrainer
        self.model = SentenceTransformer(MODEL_NAME_EMBEDDING, device="cpu")
        # prendre texte en entree et retourn un vecteur embedding 384 dimensions

    def encode(self, texts: list[str]):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

