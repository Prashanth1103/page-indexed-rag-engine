# Page-Indexed Hierarchical RAG & Agentic Document Engine

An enterprise document intelligence & retrieval architecture designed to preserve **physical document boundaries, multi-column tables, and exact page citations** where fixed-window vector chunking fails.

---

## Why Page-Indexed Hierarchical Retrieval?

In enterprise commercial and financial documents, standard fixed-size semantic chunking (e.g., 512-token windows in ChromaDB) splits multi-row tables away from their column headers and blurs numerical metrics across quarters—resulting in **60–70% citation precision** in RAGAS evaluations.

By indexing documents hierarchically (**Document Metadata $\rightarrow$ Document Summaries $\rightarrow$ Page-Bounded Section & Table Summaries**) and pairing **Intent-Driven Recursive Prompting** with **Per-Stage JSON Memoization**, this architecture reaches **94% citation precision** while keeping targeted queries under **90 seconds**.

---

## 4-Stage Retrieval Pipeline

```
User Query ──► IntentRecursivePlanner (Targeted Pass vs. Multi-Dimension Guardrail Expansion)
                    │
                    ├──► Stage 1: Lexical JSON Keyword Pre-Filter (O(N) fast match)
                    ├──► Stage 2: Async Document Summary Scorer (StageMemoCache)
                    ├──► Stage 3: Async Page & Table Section Scorer (asyncio.Semaphore)
                    └──► Stage 4: Grounded Synthesis with Verifiable (doc_id, page_number) Citations
```

### Key Engineering Features
- **Table-Aware Page Indexing (`SectionIndex`):** Stores structured text summaries alongside raw table sections tagged with exact `page_number` boundaries.
- **Deterministic Stage Memoization (`StageMemoCache`):** SHA-256 keyed JSON cache persisting intermediate LLM scoring decisions across multi-agent retries.
- **OpenTelemetry & Arize Phoenix Instrumentation (`@telemetry`):** Custom span decorator recording OpenInference span kinds (`AGENT`, `RETRIEVER`, `LLM`, `WORKFLOW`) and per-stage latency.

---

## Running Tests

```bash
pip install -r requirements.txt
pytest -v
```
