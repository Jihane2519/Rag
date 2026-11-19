# from __future__ import annotations
# from pydantic import (
#     BaseModel,
#     Field,
#     model_validator,
#     field_validator,
#     ConfigDict
# )
# from typing import List
# from uuid import uuid4


# # single source of information
# class MinimalSource(BaseModel):
#     file_path: str = Field(min_length=1)
#     first_character_index: int = Field(ge=0, strict=True)
#     last_character_index: int = Field(gt=0, strict=True)

#     @model_validator(mode="after")
#     def validate_character_indexes(self) -> MinimalSource:
#         if self.first_character_index >= self.last_character_index:
#             raise ValueError(
#                 "last_character_index must be greater than "
#                 "first_character_index"
#             )
#         return self


# class UnansweredQuestion(BaseModel):
#     model_config = ConfigDict(extra="forbid")
#     question_id: str = Field(
#         default_factory=lambda: str(uuid4()),
#         min_length=1
#     )
#     question: str = Field(min_length=1)

#     @field_validator("question")
#     @classmethod
#     def validate_query(cls, value: str) -> str:
#         value = value.strip()

#         if not value:
#             raise ValueError("query must not be empty")
#         return value


# # citation(source) et la reponse
# class AnsweredQuestion(UnansweredQuestion):
#     sources: List[MinimalSource] = Field(min_length=1)
#     answer: str = Field(min_length=1)


# # dataset of RAG questions
# class RagDataset(BaseModel):
#     rag_questions: List[AnsweredQuestion | UnansweredQuestion]


# # SINGLE QUERY WITH HIS RETRIEVED_SOURCE
# class MinimalSearchResults(BaseModel):
#     question_id: str = Field(min_length=1)
#     question: str = Field(min_length=1)
#     retrieved_sources: List[MinimalSource]


# class MinimalAnswer(MinimalSearchResults):
#     answer: str = Field(min_length=1)


# class StudentSearchResults(BaseModel):
#     search_results: List[MinimalSearchResults]
#     k: int = Field(gt=0, strict=True)


# class StudentSearchResultsAndAnswer(BaseModel):
#     search_results: List[MinimalAnswer]
#     k: int = Field(gt=0, strict=True)
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
