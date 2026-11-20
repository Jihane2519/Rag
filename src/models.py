from pydantic import BaseModel, Field
from typing import List
from uuid import uuid4


# single source of information
class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str


# citation(source) et la reponse
class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str


# dataset of RAG questions
class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


# SINGLE QUERY WITH HIS RETRIEVED_SOURCE
class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int
