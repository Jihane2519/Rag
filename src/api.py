from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI
from src.models import MinimalSource
from src.retriever import search_re
from src.embedder import Embedder
from src.indexer import load_index
from src.generator import Generator


app = FastAPI(
    title="RAG API",
    description="Local HTTP API for the RAG system",
    version="0.1.0"
)


class SearchRequest(BaseModel):
    """Validate and store incoming search and answer requests."""
    query: str = Field(..., min_length=1)
    k: int = Field(default=5, gt=0, strict=True)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Strip whitespace and reject empty queries."""
        value = value.strip()

        if not value:
            raise ValueError("query must not be empty")
        return value


embedder = Embedder()
index = load_index()
generator = Generator()


@app.post("/search", response_model=list[MinimalSource])
def search(request: SearchRequest) -> list[MinimalSource]:
    """Retrieve the most relevant sources for a search query."""
    results = search_re(
        request.query,
        k=request.k,
        embedder=embedder,
        index=index
    )
    return results


@app.post("/answer", response_model=str)
def answer(request: SearchRequest) -> str:
    """Generate an answer using the sources retrieved for a query."""
    _, all_chunks, texts, _ = index

    chunks = search_re(
        request.query,
        request.k,
        embedder,
        index
    )
    response = generator.generate_answer(
        request.query,
        chunks,
        texts,
        all_chunks
    )
    return response
