# Page-Indexed Hierarchical RAG & CrewAI Deep Researcher

An enterprise document intelligence & multi-agent research engine built with **CrewAI**, **OpenAI GPT-4o (Stateless Structured Outputs)**, **FastAPI**, and **OpenTelemetry / Arize Phoenix**.

Designed to preserve **physical document boundaries, multi-column financial/SKU tables, and verifiable page-level citations** where standard fixed-window vector chunking fails.

---

## Benchmark: Page-Indexed RAG vs. Standard ChromaDB Chunking

Evaluated on a 50-prompt enterprise commercial intelligence test suite (**28 targeted queries**, **22 deliberately vague full-corpus stress-test prompts**) under identical constraints (**single `gpt-4o` model, stateless fresh session per LLM call**):

| Pipeline Architecture | RAGAS Citation Precision | Targeted Query Latency | Full-Corpus Stress Latency |
| :--- | :---: | :---: | :---: |
| **ChromaDB Baseline (Tuned 512-Token Chunking)** | 60% – 70% | ~35s | ~2.5 min |
| **Page-Indexed Hierarchical RAG + Stage Memoization** | **94%** | **< 90s (< 1.5 min)** | **~8 min** |

---

## End-to-End Architecture

```
User Query ──► CrewAI Orchestrator (DeepResearcherCrew)
                    │
                    ├──► 1. Planner Agent (Stateless GPT-4o QueryDecompositionOutput)
                    │         └─ Targeted pass-through vs. Multi-dimension guardrail expansion
                    │
                    ├──► 2. Retrieval Agent (@tool: page_indexed_hierarchical_search)
                    │         ├─ Stage 1: Lexical JSON Keyword Pre-Filter (O(N) fast match)
                    │         ├─ Stage 2: Async GPT-4o Document Summary Scorer (StageMemoCache)
                    │         └─ Stage 3: Async GPT-4o Page & Table Section Scorer (asyncio.Semaphore)
                    │
                    └──► 3. Verification & Synthesis Agent (SynthesizedAnswerOutput)
                              └─ Enforces exact (doc_id, page_number, section_id) Pydantic citations
```

### Repository Structure
- `rag_engine/crew_workflow.py` — **CrewAI Multi-Agent Workflow** (`DeepResearcherCrew`: Planner Agent, Retrieval Specialist Agent, Citation Auditor Agent) with `@tool` integration and `output_pydantic=ResearchReport`.
- `rag_engine/llm_client.py` — **Stateless OpenAI `AsyncOpenAI` (`gpt-4o`) Client** using `beta.chat.completions.parse` structured Pydantic outputs for query decomposition, section scoring, and grounded synthesis.
- `rag_engine/retriever.py` — **4-Stage Hierarchical Page Retriever** with `asyncio.Semaphore` concurrency control and per-stage memoization.
- `rag_engine/ingest.py` — **PDF & Table Page Ingestor (`pypdf`)** extracting page-bounded sections and table summaries.
- `rag_engine/memoization.py` — **Deterministic SHA-256 Disk-Backed Stage Cache** eliminating redundant LLM calls across recursive agent steps.
- `rag_engine/phoenix_tracing.py` & `rag_engine/telemetry.py` — **OpenTelemetry + Arize Phoenix (`phoenix.otel.register`)** distributed span instrumentation (`@telemetry`).
- `rag_engine/evaluator.py` — **RAGAS & Citation Precision Evaluation Suite**.

---

## Quickstart

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
export PHOENIX_COLLECTOR_ENDPOINT="http://localhost:6006/v1/traces"

# Run unit & citation precision test suite
pytest -v

# Start FastAPI research service
uvicorn rag_engine.api:app --port 8001
```
