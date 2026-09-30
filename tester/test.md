# RAG Testing Document

This document is used to test the RAG pipeline.

## BM25

BM25 is a lexical retrieval method used to find relevant documents or chunks.

It compares terms from a user query with terms appearing in the indexed text. BM25 assigns a relevance score to each candidate chunk.

A higher BM25 score generally means that the chunk is more relevant to the query.

## Chunking

Chunking divides large documents into smaller pieces.

Each chunk should keep useful information from the original document. The chunk size can be configured with the `max_chunk_size` parameter.

## Retrieval

The retrieval stage receives a user question and searches the indexed chunks.

The system returns the top-k most relevant chunks. For example, if `k=5`, the system returns up to five relevant chunks.

## Generation

After retrieval, the selected chunks are given to the language model as context.

The model uses this context to generate a grounded answer.

If the indexed data does not contain enough information to answer the question, the system should avoid inventing information.