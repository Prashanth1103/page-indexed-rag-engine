import os
from typing import List, Optional
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from rag_engine.schemas import CitedAssertion, DocumentIndex, ResearchReport, SectionIndex
from rag_engine.telemetry import telemetry


class QueryDecompositionOutput(BaseModel):
    is_vague: bool = Field(description="True if the user prompt is broad/ambiguous and requires multi-dimensional expansion.")
    subqueries: List[str] = Field(description="Intent-aligned sub-queries covering pricing, volume, and margin dimensions.")


class RelevanceScoreOutput(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score between 0.0 and 1.0")
    rationale: str = Field(description="Concise justification for why this document or section matches the query intent.")


class SynthesizedAnswerOutput(BaseModel):
    answer_summary: str = Field(description="Comprehensive, evidence-backed response synthesizing only the provided page sections.")
    citations: List[CitedAssertion] = Field(description="Exact page-level citations supporting every claim.")


class StatelessGPT4oClient:
    """
    Stateless OpenAI GPT-4o client enforcing fresh session isolation per call
    and Pydantic Structured Outputs (`beta.chat.completions.parse`).
    Falls back to deterministic lexical scoring when OPENAI_API_KEY is not set (e.g., CI unit tests).
    """

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client: Optional[AsyncOpenAI] = AsyncOpenAI(api_key=self.api_key) if self.api_key else None

    @property
    def is_live(self) -> bool:
        return self._client is not None

    @telemetry(span_name="llm.gpt4o.decompose_query", layer="llm")
    async def decompose_query(self, query: str, required_dimensions: tuple[str, ...]) -> List[str]:
        if not self._client:
            tokens = {t.lower().strip("?,.:;") for t in query.split() if len(t) > 2}
            if len(tokens) >= 5:
                return [query.strip()]
            return [f"{query.strip()} — focus: {dim}" for dim in required_dimensions]

        completion = await self._client.beta.chat.completions.parse(
            model=self.model,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an enterprise research query planner. Every call is a fresh stateless session. "
                        "If the user prompt is targeted and specific, return it as a single subquery. "
                        f"If it is vague or broad, expand it into targeted subqueries covering: {', '.join(required_dimensions)}."
                    ),
                },
                {"role": "user", "content": query},
            ],
            response_format=QueryDecompositionOutput,
        )
        parsed = completion.choices[0].message.parsed
        return parsed.subqueries if parsed and parsed.subqueries else [query.strip()]

    @telemetry(span_name="llm.gpt4o.score_summary", layer="llm")
    async def score_summary(self, query: str, title: str, summary_text: str) -> float:
        if not self._client:
            q_terms = {w.lower().strip("?,.:;") for w in query.split() if len(w) > 2}
            s_terms = {w.lower().strip("?,.:;") for w in (title + " " + summary_text).split()}
            overlap = len(q_terms & s_terms) / max(len(q_terms), 1)
            return round(min(1.0, overlap * 1.6), 4)

        completion = await self._client.beta.chat.completions.parse(
            model=self.model,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Score how relevant the following document or section summary (including table summaries) "
                        "is to answering the user query. Return a calibrated score in [0.0, 1.0]."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\nTitle/Heading: {title}\nSummary: {summary_text}",
                },
            ],
            response_format=RelevanceScoreOutput,
        )
        parsed = completion.choices[0].message.parsed
        return round(float(parsed.score), 4) if parsed else 0.0

    @telemetry(span_name="llm.gpt4o.synthesize_report", layer="llm")
    async def synthesize_with_citations(
        self,
        query: str,
        subqueries: List[str],
        ranked_sections: List[tuple[float, SectionIndex, str]],
        cache_hits: int,
    ) -> ResearchReport:
        if not ranked_sections:
            return ResearchReport(
                query=query,
                decomposed_subqueries=subqueries,
                answer_summary="No grounded evidence met citation threshold.",
                citations=[],
                cache_hits=cache_hits,
            )

        if not self._client:
            citations = [
                CitedAssertion(
                    statement=sec.raw_content,
                    doc_id=doc_id,
                    page_number=sec.page_number,
                    section_id=sec.section_id,
                    relevance_score=sc,
                )
                for sc, sec, doc_id in ranked_sections
            ]
            summary = " | ".join(f"[{c.doc_id}:p.{c.page_number}] {c.statement}" for c in citations)
            return ResearchReport(
                query=query,
                decomposed_subqueries=subqueries,
                answer_summary=summary,
                citations=citations,
                cache_hits=cache_hits,
            )

        context_blocks = []
        for sc, sec, doc_id in ranked_sections:
            context_blocks.append(
                f"DOC_ID: {doc_id} | PAGE: {sec.page_number} | SECTION: {sec.section_id} "
                f"| SCORE: {sc}\nCONTENT: {sec.raw_content}"
            )
        joined_context = "\n\n---\n\n".join(context_blocks)

        completion = await self._client.beta.chat.completions.parse(
            model=self.model,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Synthesize a rigorous answer strictly from the provided page-indexed sections. "
                        "Every factual or numerical assertion must include its exact (doc_id, page_number, section_id) citation. "
                        "Never extrapolate beyond the provided evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": f"User Query: {query}\n\nVerified Evidence Sections:\n{joined_context}",
                },
            ],
            response_format=SynthesizedAnswerOutput,
        )
        parsed = completion.choices[0].message.parsed
        return ResearchReport(
            query=query,
            decomposed_subqueries=subqueries,
            answer_summary=parsed.answer_summary if parsed else "",
            citations=parsed.citations if parsed else [],
            cache_hits=cache_hits,
        )
