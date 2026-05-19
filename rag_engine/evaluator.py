from typing import List, Set, Tuple
from rag_engine.schemas import ResearchReport


def evaluate_citation_precision(
    report: ResearchReport,
    ground_truth_pages: Set[Tuple[str, int]],
) -> float:
    """
    Computes Citation Precision against ground-truth (doc_id, page_number) pairs:
      Precision = |Retrieved Cited Pages ∩ Ground Truth Pages| / |Retrieved Cited Pages|
    """
    if not report.citations:
        return 0.0
    hits = sum(1 for c in report.citations if (c.doc_id, c.page_number) in ground_truth_pages)
    return round(hits / len(report.citations), 4)
