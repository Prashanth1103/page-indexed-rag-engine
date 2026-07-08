import asyncio
from typing import List, Optional, Sequence, Tuple
from rag_engine.llm_client import StatelessGPT4oClient
from rag_engine.memoization import StageMemoCache
from rag_engine.schemas import DocumentIndex, ResearchReport, SectionIndex
from rag_engine.telemetry import telemetry


class IntentRecursivePlanner:
    """Decomposes broad or ambiguous user queries into bounded, intent-aligned sub-queries."""

    REQUIRED_DIMENSIONS = ("pricing_assumptions", "volume_drivers", "margin_impact")

    def __init__(self, llm_client: Optional[StatelessGPT4oClient] = None) -> None:
        self.llm = llm_client or StatelessGPT4oClient()

    @telemetry(span_name="planner.decompose_query", layer="agent")
    def decompose(self, query: str) -> List[str]:
        tokens = {t.lower().strip("?,.:;") for t in query.split() if len(t) > 2}
        if len(tokens) >= 5:
            return [query.strip()]
        return [f"{query.strip()} — focus: {dim}" for dim in self.REQUIRED_DIMENSIONS]

    @telemetry(span_name="planner.decompose_query_async", layer="agent")
    async def decompose_async(self, query: str) -> List[str]:
        return await self.llm.decompose_query(query, self.REQUIRED_DIMENSIONS)


class HierarchicalPageRetriever:
    """
    4-Stage Page-Indexed Retrieval Pipeline:
      Stage 1: Lexical JSON keyword pre-filtering across document metadata.
      Stage 2: Async GPT-4o document-summary relevancy scoring for unmatched candidates.
      Stage 3: Async GPT-4o section & table-summary relevancy scoring with page tracking.
      Stage 4: Evidence consolidation with verifiable page citations via GPT-4o Structured Outputs.
    """

    def __init__(
        self,
        corpus: Sequence[DocumentIndex],
        cache: StageMemoCache | None = None,
        llm_client: StatelessGPT4oClient | None = None,
        concurrency_limit: int = 8,
    ) -> None:
        self.corpus = list(corpus)
        self.cache = cache or StageMemoCache()
        self.llm = llm_client or StatelessGPT4oClient()
        self.planner = IntentRecursivePlanner(llm_client=self.llm)
        self._sem = asyncio.Semaphore(concurrency_limit)

    @telemetry(span_name="stage1.lexical_prefilter", layer="retriever")
    def _stage1_keyword_match(self, query: str) -> Tuple[List[DocumentIndex], List[DocumentIndex]]:
        q_terms = {w.lower().strip("?,.:;") for w in query.split() if len(w) > 2}
        matched, remaining = [], []
        for doc in self.corpus:
            doc_kw = {k.lower() for k in doc.keywords}
            if q_terms & doc_kw:
                matched.append(doc)
            else:
                remaining.append(doc)
        return matched, remaining

    @telemetry(span_name="stage2.async_doc_summary_score", layer="llm")
    async def _score_doc_summary(self, query: str, doc: DocumentIndex) -> float:
        payload = {"q": query, "doc_id": doc.doc_id, "summary": doc.doc_summary}
        cached = self.cache.get("doc_score", payload)
        if cached is not None:
            return float(cached)

        async with self._sem:
            score = await self.llm.score_summary(query, doc.title, doc.doc_summary)
            self.cache.set("doc_score", payload, score)
            return score

    @telemetry(span_name="stage3.async_section_score", layer="llm")
    async def _score_section(self, query: str, doc_id: str, section: SectionIndex) -> Tuple[float, SectionIndex, str]:
        payload = {"q": query, "doc_id": doc_id, "sec_id": section.section_id}
        cached = self.cache.get("sec_score", payload)
        if cached is not None:
            return float(cached), section, doc_id

        async with self._sem:
            summary_input = f"{section.summary} Keywords: {' '.join(section.keywords)}"
            score = await self.llm.score_summary(query, section.heading, summary_input)
            self.cache.set("sec_score", payload, score)
            return score, section, doc_id

    @telemetry(span_name="workflow.execute_research", layer="workflow")
    async def retrieve_and_synthesize(
        self,
        query: str,
        doc_threshold: float = 0.15,
        section_threshold: float = 0.20,
        top_k: int = 5,
    ) -> ResearchReport:
        subqueries = await self.planner.decompose_async(query)
        candidate_docs_map = {}

        for sq in subqueries:
            matched, remaining = self._stage1_keyword_match(sq)
            for d in matched:
                candidate_docs_map[d.doc_id] = d

            if remaining:
                scores = await asyncio.gather(*(self._score_doc_summary(sq, d) for d in remaining))
                for d, sc in zip(remaining, scores):
                    if sc >= doc_threshold:
                        candidate_docs_map[d.doc_id] = d

        section_tasks = []
        for sq in subqueries:
            for doc in candidate_docs_map.values():
                for sec in doc.sections:
                    section_tasks.append(self._score_section(sq, doc.doc_id, sec))

        scored_sections = await asyncio.gather(*section_tasks) if section_tasks else []
        best_by_sec = {}
        for sc, sec, doc_id in scored_sections:
            if sc >= section_threshold:
                key = (doc_id, sec.section_id)
                if key not in best_by_sec or sc > best_by_sec[key][0]:
                    best_by_sec[key] = (sc, sec, doc_id)

        ranked = sorted(best_by_sec.values(), key=lambda x: x[0], reverse=True)[:top_k]
        return await self.llm.synthesize_with_citations(
            query=query,
            subqueries=subqueries,
            ranked_sections=ranked,
            cache_hits=self.cache.hits,
        )
