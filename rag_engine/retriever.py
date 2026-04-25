import asyncio
from typing import Callable, List, Sequence, Tuple
from rag_engine.memoization import StageMemoCache
from rag_engine.schemas import CitedAssertion, DocumentIndex, ResearchReport, SectionIndex
from rag_engine.telemetry import telemetry


class IntentRecursivePlanner:
    """Decomposes broad or ambiguous user queries into bounded, intent-aligned sub-queries."""

    REQUIRED_DIMENSIONS = ("pricing_assumptions", "volume_drivers", "margin_impact")

    @telemetry(span_name="planner.decompose_query", layer="agent")
    def decompose(self, query: str) -> List[str]:
        tokens = {t.lower().strip("?,.:;") for t in query.split() if len(t) > 2}
        if len(tokens) >= 5:
            return [query.strip()]
        # Vague query guardrail: expand across core commercial dimensions
        return [f"{query.strip()} — focus: {dim}" for dim in self.REQUIRED_DIMENSIONS]


class HierarchicalPageRetriever:
    """
    4-Stage Page-Indexed Retrieval Pipeline:
      Stage 1: Lexical JSON keyword pre-filtering across document metadata.
      Stage 2: Async document-summary relevancy scoring for unmatched candidates.
      Stage 3: Async section & table-summary relevancy scoring with page tracking.
      Stage 4: Evidence consolidation with verifiable page citations.
    """

    def __init__(
        self,
        corpus: Sequence[DocumentIndex],
        cache: StageMemoCache | None = None,
        concurrency_limit: int = 8,
    ) -> None:
        self.corpus = list(corpus)
        self.cache = cache or StageMemoCache()
        self.planner = IntentRecursivePlanner()
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
            await asyncio.sleep(0.002)
            q_terms = {w.lower().strip("?,.:;") for w in query.split() if len(w) > 2}
            s_terms = {w.lower().strip("?,.:;") for w in doc.doc_summary.split()}
            overlap = len(q_terms & s_terms) / max(len(q_terms), 1)
            score = round(min(1.0, overlap * 1.5), 4)
            self.cache.set("doc_score", payload, score)
            return score

    @telemetry(span_name="stage3.async_section_score", layer="llm")
    async def _score_section(self, query: str, doc_id: str, section: SectionIndex) -> Tuple[float, SectionIndex, str]:
        payload = {"q": query, "doc_id": doc_id, "sec_id": section.section_id}
        cached = self.cache.get("sec_score", payload)
        if cached is not None:
            return float(cached), section, doc_id

        async with self._sem:
            await asyncio.sleep(0.002)
            q_terms = {w.lower().strip("?,.:;") for w in query.split() if len(w) > 2}
            sec_terms = {
                w.lower().strip("?,.:;")
                for w in (section.summary + " " + " ".join(section.keywords)).split()
            }
            overlap = len(q_terms & sec_terms) / max(len(q_terms), 1)
            score = round(min(1.0, overlap * 1.8), 4)
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
        subqueries = self.planner.decompose(query)
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
        citations = [
            CitedAssertion(
                statement=sec.raw_content,
                doc_id=doc_id,
                page_number=sec.page_number,
                section_id=sec.section_id,
                relevance_score=sc,
            )
            for sc, sec, doc_id in ranked
        ]
        summary = " | ".join(f"[{c.doc_id}:p.{c.page_number}] {c.statement}" for c in citations)
        return ResearchReport(
            query=query,
            decomposed_subqueries=subqueries,
            answer_summary=summary or "No grounded evidence met citation threshold.",
            citations=citations,
            cache_hits=self.cache.hits,
        )
