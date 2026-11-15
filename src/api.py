from pydantic import BaseModel, Field
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
    query: str = Field(..., min_length=1)
    k: int = Field(default=5, gt=0)


embedder = Embedder()
index = load_index()
generator = Generator()


@app.post("/search", response_model=list[MinimalSource])
def search(request: SearchRequest) -> list[MinimalSource]:
    results = search_re(
        request.query,
        k=request.k,
        embedder=embedder,
        index=index
    )
    return results


@app.post("/answer", response_model=str)
def answer(request: SearchRequest) -> str:
    _, all_chunks, texts, _ = index

    chunks = search_re(
        request.query,
        request.k,
        embedder,
        index
    )
    print("QUESTION:", request.query)
    print("RETRIEVED CHUNKS:", chunks)

    context_texts = [
        texts[all_chunks.index(chunk)]
        for chunk in chunks
    ]

    print("CONTEXT TEXTS:", context_texts)
    print("\n========== DEBUG RETRIEVAL ==========")
    print("QUESTION:", request.query)

    for rank, chunk in enumerate(chunks, start=1):
        print(f"\n--- RESULT {rank} ---")
        print("File:", chunk.file_path)
        print("Start:", chunk.first_character_index)
        print("End:", chunk.last_character_index)

        for i, c in enumerate(all_chunks):
            if (
                c.file_path == chunk.file_path
                and c.first_character_index == chunk.first_character_index
            ):
                print("TEXT:", texts[i])
                break

    print("=====================================\n")
    response = generator.generate_answer(
        request.query,
        chunks,
        texts,
        all_chunks
    )
    print("QUESTION:", request.query)
    print("RETRIEVED CHUNKS:", chunks)
    print("CONTEXT TEXTS:", [texts[all_chunks.index(c)] for c in chunks])

    return response
