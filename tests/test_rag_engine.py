import shutil
import tempfile
import pytest
from rag_engine.evaluator import evaluate_citation_precision
from rag_engine.memoization import StageMemoCache
from rag_engine.retriever import HierarchicalPageRetriever
from rag_engine.schemas import DocumentIndex, SectionIndex
from rag_engine.telemetry import exporter


@pytest.fixture
def sample_corpus():
    return [
        DocumentIndex(
            doc_id="RGM_2026_Q2",
            title="India RGM SKU Elasticity & MACO Report Q2",
            doc_summary="Analysis of beer SKU price elasticity, VAT deductibles, and MACO margins across Karnataka and Maharashtra.",
            keywords=["maco", "elasticity", "sku", "karnataka"],
            sections=[
                SectionIndex(
                    section_id="sec_table_1",
                    page_number=14,
                    heading="Premium SKU Price Elasticity Table",
                    summary="Table showing SKU_501 price elasticity of -1.18 and 4.2% MACO expansion in Q2.",
                    is_table=True,
                    raw_content="SKU_501 exhibited price elasticity of -1.18 with +4.2% MACO expansion.",
                    keywords=["sku_501", "elasticity", "maco", "table"],
                ),
                SectionIndex(
                    section_id="sec_intro",
                    page_number=2,
                    heading="Executive Overview",
                    summary="General macroeconomic commentary for Q2 beverage sector.",
                    is_table=False,
                    raw_content="Macroeconomic conditions remained stable.",
                    keywords=["macro", "overview"],
                ),
            ],
        ),
        DocumentIndex(
            doc_id="FORECAST_2026_NAT",
            title="National Volume Forecasting Baseline",
            doc_summary="Monthly seasonal volume projections and weather sensitivity across regions.",
            keywords=["forecast", "seasonal", "summer"],
            sections=[
                SectionIndex(
                    section_id="sec_summer",
                    page_number=7,
                    heading="Summer Peak Demand Drivers",
                    summary="Temperature anomalies drive 14% volume uplift in May-June.",
                    is_table=False,
                    raw_content="Summer temperature anomalies drive +14% volume uplift.",
                    keywords=["summer", "temperature", "uplift"],
                )
            ],
        ),
    ]


@pytest.mark.asyncio
async def test_hierarchical_retrieval_and_citation_precision(sample_corpus):
    exporter.clear()
    tmp_dir = tempfile.mkdtemp()
    try:
        cache = StageMemoCache(cache_dir=tmp_dir)
        retriever = HierarchicalPageRetriever(sample_corpus, cache=cache)

        report = await retriever.retrieve_and_synthesize(
            "What was the SKU_501 price elasticity and MACO margin expansion in Q2?",
            top_k=1,
        )
        assert len(report.citations) == 1
        assert report.citations[0].doc_id == "RGM_2026_Q2"
        assert report.citations[0].page_number == 14

        precision = evaluate_citation_precision(report, {("RGM_2026_Q2", 14)})
        assert precision == 1.0

        # Second identical query hits disk stage memoization cache
        report_cached = await retriever.retrieve_and_synthesize(
            "What was the SKU_501 price elasticity and MACO margin expansion in Q2?",
            top_k=1,
        )
        assert report_cached.cache_hits > 0

        spans = exporter.get_finished_spans()
        assert any(s.name == "workflow.execute_research" for s in spans)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
