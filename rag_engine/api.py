from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from rag_engine.retriever import HierarchicalPageRetriever
from rag_engine.schemas import DocumentIndex, ResearchReport, SectionIndex

app = FastAPI(
    title="Deep Researcher — Page-Indexed Agentic RAG API",
    version="0.2.0",
    description="Enterprise multi-agent document extraction and page-indexed RAG service.",
)

_DEFAULT_CORPUS: List[DocumentIndex] = [
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
            )
        ],
    )
]

_retriever = HierarchicalPageRetriever(_DEFAULT_CORPUS)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/v1/research", response_model=ResearchReport)
async def execute_research_endpoint(req: QueryRequest) -> ResearchReport:
    return await _retriever.retrieve_and_synthesize(req.query, top_k=req.top_k)
