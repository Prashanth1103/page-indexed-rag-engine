import re
from pathlib import Path
from typing import List
from pypdf import PdfReader
from rag_engine.schemas import DocumentIndex, SectionIndex
from rag_engine.telemetry import telemetry


class PageIndexedIngestor:
    """
    Ingests PDFs or structured markdown reports into page-indexed DocumentIndex objects,
    preserving page boundaries and generating searchable summaries for embedded tables.
    """

    TABLE_PATTERN = re.compile(r"(\|.+\|\n\|[-:| ]+\|)", re.MULTILINE)

    @telemetry(span_name="ingest.pdf_document", layer="ingestor")
    def ingest_pdf(self, pdf_path: str | Path, doc_id: str, title: str) -> DocumentIndex:
        reader = PdfReader(str(pdf_path))
        sections: List[SectionIndex] = []
        doc_keywords: set[str] = set()

        for idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            has_table = bool(self.TABLE_PATTERN.search(text)) or ("table" in text.lower())
            words = [
                w.lower().strip(".,:;()[]")
                for w in text.split()
                if len(w) > 4 and w.isalpha()
            ]
            top_kw = list(dict.fromkeys(words))[:12]
            doc_keywords.update(top_kw[:5])

            first_line = text.splitlines()[0][:80] if text.splitlines() else f"Page {idx}"
            summary = text[:280].replace("\n", " ")

            sections.append(
                SectionIndex(
                    section_id=f"{doc_id}_p{idx}",
                    page_number=idx,
                    heading=first_line,
                    summary=f"[Table Summary] {summary}" if has_table else summary,
                    is_table=has_table,
                    raw_content=text,
                    keywords=top_kw,
                )
            )

        doc_summary = " ".join(s.summary for s in sections[:3])[:500]
        return DocumentIndex(
            doc_id=doc_id,
            title=title,
            doc_summary=doc_summary,
            keywords=sorted(doc_keywords)[:25],
            sections=sections,
        )
