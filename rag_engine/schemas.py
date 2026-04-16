from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SectionIndex(BaseModel):
    section_id: str
    page_number: int = Field(..., ge=1)
    heading: str
    summary: str
    is_table: bool = False
    raw_content: str
    keywords: List[str] = Field(default_factory=list)


class DocumentIndex(BaseModel):
    doc_id: str
    title: str
    doc_summary: str
    keywords: List[str] = Field(default_factory=list)
    sections: List[SectionIndex] = Field(default_factory=list)


class CitedAssertion(BaseModel):
    statement: str
    doc_id: str
    page_number: int
    section_id: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class ResearchReport(BaseModel):
    query: str
    decomposed_subqueries: List[str]
    answer_summary: str
    citations: List[CitedAssertion]
    cache_hits: int = 0
