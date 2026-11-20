*This project has been created as part of the 42 curriculum by jbounou.*

# RAG Against the Machine

## Description

**RAG Against the Machine** is a Retrieval-Augmented Generation (RAG) project developed as part of the 42 curriculum.

The goal of this project is to build a system capable of answering questions about a given collection of documents and source code.

Instead of relying only on the knowledge stored in a language model, the system first retrieves relevant information from a local knowledge base. The retrieved content is then provided to a language model, which generates an answer based on that context.

The project combines:

- Document loading and preprocessing.
- Document chunking.
- Keyword-based retrieval using BM25.
- Semantic retrieval using embeddings.
- Retrieval result ranking.
- Answer generation using a Large Language Model (LLM).
- Dataset search and evaluation.

## System Architecture

The RAG pipeline consists of several components that work together to retrieve relevant information and generate answers.

### 1. Document Loading

The system loads supported files from the configured data directory.

Supported file formats include:

- Python (`.py`)
- Markdown (`.md`)
- Text (`.txt`)

The loader recursively explores directories and extracts file contents.

### 2. Chunking

Documents are divided into smaller text segments called chunks.

Each chunk keeps its source file path and character positions so that retrieved information can be traced back to its original location.

### 3. Indexing

The indexing process prepares the documents for retrieval.

It includes:

- Tokenizing chunks for BM25 indexing.
- Generating semantic embeddings.
- Storing processed data for future searches.

The generated index data is saved locally so that it can be loaded without rebuilding the entire index every time.

### 4. Retrieval

When a user submits a question, the system searches the indexed chunks using two complementary methods:

- BM25 for keyword-based retrieval.
- Embedding similarity for semantic retrieval.

The results are combined and ranked to identify the most relevant chunks.

### 5. Generation

The retrieved chunks are passed to the language model together with the user's question.

The model used in this project is **Qwen3-0.6B**.

The generator uses the retrieved context to produce an answer to the question.

### RAG Pipeline Overview

```text
             Documents
                 |
                 v
          Document Loader
                 |
                 v
             Chunking
                 |
                 v
              Indexing
                 |
          +------+------+
          |             |
          v             v
       BM25          Embeddings
          |             |
          v             v
     Keyword Search  Semantic Search
          |             |
          +------+------+
                 |
                 v
          Result Ranking
                 |
                 v
          Retrieved Chunks
                 |
                 v
          Question + Context
                 |
                 v
           Qwen3-0.6B
                 |
                 v
             Answer
```

## Chunking Strategy

Chunking is the process of dividing large documents into smaller, manageable text segments.

This project uses different splitting strategies depending on the file type:

- **Python files:** Code-aware chunking is used to preserve the structure of source code as much as possible.
- **Markdown files:** Markdown-aware chunking is used to organize content according to document structure.
- **Text files:** Text is divided into smaller segments.

The default maximum chunk size is configured in the project settings.

Each chunk stores metadata including:

- `file_path`: the original file location.
- `first_character_index`: the starting character position.
- `last_character_index`: the ending character position.

This metadata allows the system to identify the exact source of retrieved information.

## Retrieval Method

The project uses a hybrid retrieval approach combining BM25 and semantic search.

### BM25 Retrieval

BM25 is a keyword-based ranking algorithm.

It evaluates the relevance of a document to a query by considering factors such as:

- Term frequency.
- Inverse document frequency.
- Document length.

BM25 is useful when the question contains specific keywords, identifiers, function names, or technical terms.

### Semantic Retrieval

Semantic retrieval uses text embeddings to represent queries and document chunks as numerical vectors.

This project uses the `sentence-transformers/all-MiniLM-L6-v2` model to generate embeddings.

The similarity between a query embedding and chunk embeddings is calculated to identify semantically related content.

### Hybrid Ranking

The system combines keyword-based and semantic retrieval results to benefit from both approaches.

This helps retrieve relevant information even when the question does not use exactly the same words as the source documents.

The final results are ranked, and the top `k` chunks are returned.

## Performance Analysis

The retrieval system is evaluated using Recall@k.

Recall@k measures whether the relevant source documents are included among the top `k` retrieved results.

The project uses source file paths and character-range overlap to determine whether a retrieved chunk matches a reference source.

### Recall@5 Results

The retrieval system was evaluated on two datasets using Recall@5.

| Dataset | Recall@5 | Target |
|---|---:|---:|
| Documentation | 84.0% | 80% |
| Code | 60.6% | 50% |

Both datasets exceeded their respective evaluation targets.

### Performance Considerations

The project includes several mechanisms to improve efficiency:

- Local index storage avoids rebuilding the index for every search.
- Incremental indexing processes modified files instead of rebuilding all chunks unnecessarily.
- Query caching avoids repeating retrieval work for identical queries within the same running process.
- Embeddings are reused from the saved index when available.

## Design Decisions

Several implementation choices were made during development.

### Hybrid Retrieval

BM25 and semantic search complement each other. BM25 handles exact keyword matches, while embeddings help identify conceptually related content.

### Local Index Storage

Processed chunks, BM25 data, and embeddings are stored locally to reduce repeated preprocessing.

### Incremental Indexing

The indexing system tracks file changes to avoid processing unchanged files unnecessarily.

### Pydantic Models

Pydantic is used to validate and structure the data exchanged between the different components, including questions, retrieved sources, and generated answers.

### Fire CLI

Fire provides a simple command-line interface for executing the main project operations.

### Qwen3-0.6B

A relatively small language model was selected for answer generation, making it possible to run the generation component locally.

## Challenges Faced

During development, several challenges were encountered:

### 1. Retrieval Quality

Initial retrieval results did not always contain the expected source chunks.

The retrieval pipeline was improved by combining BM25 and semantic similarity instead of relying on keyword search alone.

### 2. Document Processing

Different file types require different splitting strategies.

Separate chunking approaches were implemented for Python and Markdown documents.

### 3. Index Management

Rebuilding the complete index after every modification can be inefficient.

Incremental indexing was introduced to identify modified files and process only the necessary content.

### 4. Model Resource Management

Loading language models requires significant disk space and memory.

The Hugging Face cache was configured to use an appropriate storage location, and the model is loaded locally when available.

### 5. Evaluation

Retrieval quality needs to be measured against reference sources rather than judged only by generated answers.

Recall@k was used to evaluate whether the expected source chunks appear in the retrieved results.

## Instructions

### Requirements

- Python 3.10 or higher.
- [uv](https://docs.astral.sh/uv/) package manager.
- Git.
- Sufficient disk space and memory for the embedding and generation models.

### Installation

Clone the repository:

```bash
git clone <repository-url>
cd RAG
```

Install the project dependencies:

```bash
make install
```

### Build the Index

Build the retrieval index from the project documents:

```bash
uv run python -m src index --max_chunk_size 2000
```

### Search

Search for the most relevant chunks for an individual question:

```bash
uv run python -m src search "What is RAG"
```

### Generate an Answer

Generate an answer using the retrieved context and the language model:

```bash
uv run python -m src answer "What is RAG"
```

### Search the Documentation Dataset

Retrieve the top 5 relevant chunks for each question in the documentation dataset:

```bash
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 5 \
    --save_directory data/output/search_results/UnansweredQuestions
```

### Evaluate Documentation Retrieval

Evaluate the retrieved documentation chunks against the reference dataset:

```bash
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

### Search the Code Dataset

Retrieve the top 5 relevant chunks for each question in the code dataset:

```bash
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
    --k 5 \
    --save_directory data/output/search_results/UnansweredQuestions
```

### Evaluate Code Retrieval

Evaluate the retrieved code chunks against the reference dataset:

```bash
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json
```

### API

The project also provides a FastAPI application.

Start the development server:

```bash
uv run uvicorn src.api:app --reload
```

Open the interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

## Example Usage

The following example demonstrates how to use the RAG system.

**1. Build the index**

```bash
uv run python -m src index --max_chunk_size 2000
```

**2. Search for relevant information**

```bash
uv run python -m src search "What is RAG"
```

**3. Generate an answer**

```bash
uv run python -m src answer "What is RAG"
```

The system retrieves relevant information from the indexed documents and uses Qwen3-0.6B to generate an answer based on the retrieved context.
## Project Structure

```text
RAG/
├── data/
│   ├── datasets/
│   ├── output/
│   ├── processed/
│   │   └── manifest.json
│   └── raw/
├── moulinette/
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── api.py
│   ├── cache.py
│   ├── chunker.py
│   ├── config.py
│   ├── embedder.py
│   ├── generator.py
│   ├── indexer.py
│   ├── loader.py
│   ├── models.py
│   └── retriever.py
├── Makefile
├── pyproject.toml
└── uv.lock
```

## Resources

### Documentation and References

- [Retrieval-Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401) — Original RAG research paper.
- [Hugging Face Documentation](https://huggingface.co/docs) — Documentation for models, tokenizers, and transformers.
- [Sentence Transformers](https://www.sbert.net/) — Documentation for sentence embeddings and semantic search.
- [Rank-BM25](https://github.com/dorianbrown/rank_bm25) — BM25 implementation used for keyword retrieval.
- [Pydantic Documentation](https://docs.pydantic.dev/) — Data validation and models.
- [FastAPI Documentation](https://fastapi.tiangolo.com/) — API development.
- [Python Documentation](https://docs.python.org/3/) — Python language reference.
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — Deep learning framework.
- [Qwen Documentation](https://qwen.readthedocs.io/) — Qwen model documentation.

### AI Usage

AI tools were used as development assistance during the project.

They were used for:

- Understanding RAG concepts and retrieval techniques.
- Explaining technical documentation and Python libraries.
- Debugging errors and investigating unexpected behavior.
- Discussing possible improvements to chunking and retrieval strategies.
- Reviewing code structure and suggesting refactoring ideas.

---

**RAG Against the Machine — Will you answer my questions?**
