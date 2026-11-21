from pydantic import BaseModel, Field
from typing import List
from uuid import uuid4


# single source of information
class MinimalSource(BaseModel):
    """Represent a source file and its character positions."""
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represent a question without its answer or sources."""
    question_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str


# citation(source) et la reponse
class AnsweredQuestion(UnansweredQuestion):
    """Represent a question with its answer and reference sources."""
    sources: List[MinimalSource]
    answer: str


# dataset of RAG questions
class RagDataset(BaseModel):
    """Represent a dataset containing answered and unanswered questions."""
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


# SINGLE QUERY WITH HIS RETRIEVED_SOURCE
class MinimalSearchResults(BaseModel):
    """Store a question and its retrieved source chunks."""
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Store a question, its retrieved sources, and its generated answer."""
    answer: str


class StudentSearchResults(BaseModel):
    """Store search results for a dataset and the retrieval parameter k."""
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Store search results and generated answers for a dataset."""
    search_results: List[MinimalAnswer]
    k: int
