"""
RAG Stage 1 — Ingestion Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Imports core pipeline classes from rag_stage1_ingestion_pipeline.py.

Key design decisions vs the previous version:
  1. DirectoryLoader (from pipeline) is used directly for PDFs, markdown,
     and code — no duplicate AnalysisPDFDirectoryLoader / AnalysisMarkdownLoader.
     Analysis-specific signals (heading count, function count, etc.) are
     computed from doc.content inside run_analysis(), not baked into loaders.
     Rule: loaders enrich metadata only if it drives a PRODUCTION decision.
     If it only drives a chart, compute it here.

  2. Three custom loaders remain because DirectoryLoader cannot handle them:
       JSONFAQLoader    — structured Q&A schema, external doc_id as stable key
       CSVProductLoader — column-semantic content building, configurable columns
       WebCacheLoader   — boilerplate stripping + boilerplate_removed_pct metric

  3. Image handling is fully wired:
       pdf_image_strategy="none"    → detect + flag (default, no extra cost)
       pdf_image_strategy="ocr"     → Tesseract OCR appended to content
       pdf_image_strategy="caption" → Claude vision caption appended to content
     Switch via PDF_IMAGE_STRATEGY constant at the top of this file.

  4. Document previews — new section in the report shows a sample document
     card for each source type: id, metadata, and content preview.

Usage:
    python setup_data.py               # download test data first
    python analyze_ingestion.py        # → ingestion_analysis_report.html

Optional env vars:
    ANTHROPIC_API_KEY   required only when PDF_IMAGE_STRATEGY="caption"

Install:
    pip install pymupdf4llm httpx beautifulsoup4
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ──────────────────────────────────────────────────────────
#  Imports from pipeline — the only source of truth for
#  Document, build_metadata, BaseLoader, and all loaders.
# ──────────────────────────────────────────────────────────

try:
    from rag_stage1_ingestion_pipeline import (
        Document,
        build_metadata,
        BaseLoader,
        DirectoryLoader,      # handles PDF + markdown + code + text + config
        IngestionPipeline,
    )
except ImportError:
    print(
        "❌  Could not import rag_stage1_ingestion_pipeline.\n"
        "    Make sure both .py files are in the same directory."
    )
    sys.exit(1)

# ── Config ────────────────────────────────────────────────
ASSETS      = Path("./assets")
REPORT_PATH = Path("./ingestion_analysis_report.html")

# Change this to "ocr" or "caption" to test those strategies.
# "caption" requires ANTHROPIC_API_KEY in environment.
PDF_IMAGE_STRATEGY: str = "none"


# ══════════════════════════════════════════════════════════
#  CUSTOM LOADERS — only for source types DirectoryLoader
#  cannot handle. All three produce Documents using the
#  same build_metadata() contract from the pipeline.
# ══════════════════════════════════════════════════════════

class JSONFAQLoader(BaseLoader):
    """
    Loads a FAQ JSON corpus: {"faqs": [{"id","question","answer",...}]}.
    DirectoryLoader would ingest this as raw JSON config text, losing
    the Q&A structure. This loader builds column-semantic content and
    uses the external FAQ id as a stable doc_id for incremental re-runs.
    """

    def __init__(self, json_path: Path):
        self.json_path = json_path

    async def load(self) -> List[Document]:
        if not self.json_path.exists():
            return []

        raw  = await asyncio.to_thread(self.json_path.read_text, encoding="utf-8")
        data = json.loads(raw)
        faqs = data.get("faqs", [])

        docs    = []
        skipped = 0
        for item in faqs:
            question = (item.get("question") or "").strip()
            answer   = (item.get("answer")   or "").strip()
            if not question or not answer:
                print(f"  ⚠  FAQ {item.get('id')}: null/empty field — will be filtered")
                skipped += 1
                continue

            docs.append(Document(
                content=f"Question: {question}\nAnswer: {answer}",
                metadata=build_metadata(
                    source=str(self.json_path),
                    source_type="json_faq",
                    file_type="json",
                    faq_id=item.get("id", ""),
                    category=item.get("category", ""),
                    tags=",".join(item.get("tags", [])),
                    external_id=item.get("id", ""),
                ),
                doc_id=f"faq_{item.get('id', '')}",
            ))

        print(f"  ✓  {self.json_path.name}: {len(docs)} valid FAQs "
              f"({skipped} skipped)")
        return docs


class CSVProductLoader(BaseLoader):
    """
    Loads a product CSV with configurable column mapping.
    DirectoryLoader would produce raw CSV text — no column semantics.
    This loader builds "col: value\\n..." content that embeds column
    names so the LLM understands what each value means.
    """

    def __init__(self, csv_path: Path, content_columns: List[str],
                 metadata_columns: List[str]):
        self.csv_path      = csv_path
        self.content_cols  = content_columns
        self.metadata_cols = metadata_columns

    async def load(self) -> List[Document]:
        if not self.csv_path.exists():
            return []

        raw     = await asyncio.to_thread(self.csv_path.read_text, encoding="utf-8")
        lines   = raw.strip().splitlines()
        headers = [h.strip() for h in lines[0].split(",")]

        docs = []
        skipped = 0
        for i, line in enumerate(lines[1:], start=1):
            row = self._parse_row(line, headers)
            if not all(row.get(c, "").strip() for c in self.content_cols[:2]):
                print(f"  ⚠  Row {i}: missing required columns — skipped")
                skipped += 1
                continue

            content = "\n".join(
                f"{col}: {row[col]}"
                for col in self.content_cols
                if row.get(col, "").strip()
            )
            extra = {col: row.get(col, "") for col in self.metadata_cols}
            docs.append(Document(
                content=content,
                metadata=build_metadata(
                    source=str(self.csv_path),
                    source_type="csv",
                    file_type="csv",
                    row_index=i,
                    **extra,
                ),
            ))

        print(f"  ✓  {self.csv_path.name}: {len(docs)} rows "
              f"({skipped} skipped)")
        return docs

    @staticmethod
    def _parse_row(line: str, headers: List[str]) -> Dict[str, str]:
        values, cur, in_q = [], "", False
        for ch in line:
            if ch == '"':  in_q = not in_q
            elif ch == ',' and not in_q:
                values.append(cur.strip()); cur = ""
            else: cur += ch
        values.append(cur.strip())
        return dict(zip(headers, values + [""] * (len(headers) - len(values))))


class WebCacheLoader(BaseLoader):
    """
    Loads pre-cached HTML files with boilerplate stripping.
    DirectoryLoader has no HTML support. This loader also captures
    boilerplate_removed_pct — an analysis-specific signal that
    proves the stripping actually removed meaningful noise.
    """

    def __init__(self, directory: Path):
        self.directory = directory

    async def load(self) -> List[Document]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("pip install beautifulsoup4")

        docs = []
        for p in sorted(self.directory.glob("*.html")):
            try:
                html = await asyncio.to_thread(p.read_text, encoding="utf-8")
                soup = BeautifulSoup(html, "html.parser")
                raw_len = len(soup.get_text())

                for tag in soup.find_all(
                    ["nav", "footer", "script", "style", "aside", "header", "noscript"]
                ):
                    tag.decompose()

                clean    = soup.get_text(separator="\n", strip=True)
                bp_pct   = round((1 - len(clean) / raw_len) * 100, 1) if raw_len else 0
                title_t  = soup.find("title")
                meta_url = soup.find("meta", attrs={"name": "source-url"})

                docs.append(Document(
                    content=clean,
                    metadata=build_metadata(
                        source=meta_url["content"] if meta_url else str(p),
                        source_type="web",
                        file_type="html",
                        file_name=p.name,
                        title=title_t.string.strip() if title_t else "",
                        raw_char_count=raw_len,
                        stripped_char_count=len(clean),
                        boilerplate_removed_pct=bp_pct,
                    ),
                ))
                print(f"  ✓  {p.name}: {len(clean):,} chars "
                      f"({bp_pct}% boilerplate removed)")
            except Exception as e:
                print(f"  ✗  {p.name}: {e}")
        return docs


# ══════════════════════════════════════════════════════════
#  ANALYSIS ENGINE
#  Pure computation — no I/O.
#  Analysis-specific signals (heading_count, function_count,
#  etc.) are derived from doc.content here, not in loaders.
# ══════════════════════════════════════════════════════════

REQUIRED_METADATA = ["source", "source_type", "file_type", "ingested_at"]


@dataclass
class AnalysisResults:
    total_docs:              int              = 0
    docs_by_source:          Dict[str, int]   = field(default_factory=dict)
    avg_words_by_source:     Dict[str, float] = field(default_factory=dict)
    total_tokens_by_source:  Dict[str, int]   = field(default_factory=dict)
    short_docs:              List[Dict]        = field(default_factory=list)
    long_docs:               List[Dict]        = field(default_factory=list)
    duplicate_ids:           List[str]         = field(default_factory=list)
    duplicate_count:         int              = 0
    metadata_issues:         List[Dict]        = field(default_factory=list)
    # PDF image analysis
    pdf_image_stats:         Dict[str, Any]   = field(default_factory=dict)
    # Source-specific enrichment (derived from content)
    markdown_stats:          Dict[str, Any]   = field(default_factory=dict)
    code_stats:              Dict[str, Any]   = field(default_factory=dict)
    web_stats:               Dict[str, Any]   = field(default_factory=dict)
    # Totals
    total_estimated_tokens:  int              = 0
    projected_chunks_512:    int              = 0
    timing_by_source:        Dict[str, float] = field(default_factory=dict)
    chunk_readiness:         Dict[str, int]   = field(default_factory=dict)
    # Previews — one sample doc per source_type
    previews:                Dict[str, Dict]  = field(default_factory=dict)


def _tokens(doc: Document) -> int:
    return max(1, doc.char_count() // 4)


def _non_ascii_ratio(doc: Document) -> float:
    if not doc.content: return 0.0
    return round(sum(1 for c in doc.content if ord(c) > 127) / len(doc.content), 4)


def _derive_markdown_signals(content: str) -> Dict[str, int]:
    return {
        "heading_count":    len(re.findall(r"^#{1,6}\s", content, re.MULTILINE)),
        "code_block_count": len(re.findall(r"```", content)) // 2,
    }


def _derive_code_signals(content: str) -> Dict[str, int]:
    return {
        "function_count":  len(re.findall(r"^def\s+\w+",   content, re.MULTILINE)),
        "class_count":     len(re.findall(r"^class\s+\w+", content, re.MULTILINE)),
        "docstring_count": len(re.findall(r'"""',           content)) // 2,
    }


def run_analysis(docs: List[Document], timings: Dict[str, float]) -> AnalysisResults:
    r = AnalysisResults(total_docs=len(docs), timing_by_source=timings)

    seen_ids:      Dict[str, str]  = {}
    source_words:  Dict[str, list] = defaultdict(list)
    source_tokens: Dict[str, int]  = defaultdict(int)
    ready = not_ready = 0

    # PDF image tracking
    pdf_img_total = pdf_img_with = pdf_img_ocr_chars = pdf_img_captioned = 0

    # Markdown / code / web aggregates (derived from content)
    md_headings = md_code_blocks = 0
    code_funcs = code_classes = code_no_docstring = 0
    bp_pcts: List[float] = []

    for doc in docs:
        st     = doc.metadata.get("source_type", "unknown")
        words  = doc.word_count()
        tokens = _tokens(doc)
        nar    = _non_ascii_ratio(doc)

        source_words[st].append(words)
        source_tokens[st] += tokens
        r.total_estimated_tokens += tokens

        # ── Outliers
        if words < 20:
            r.short_docs.append({"id": doc.doc_id, "source_type": st,
                                  "words": words,
                                  "preview": doc.content[:80].replace("\n", " ")})
        if words > 2000:
            r.long_docs.append({"id": doc.doc_id, "source_type": st,
                                 "words": words,
                                 "source": Path(doc.metadata.get("source","")).name})

        # ── Dedup
        if doc.doc_id in seen_ids:
            r.duplicate_ids.append(doc.doc_id)
            r.duplicate_count += 1
        else:
            seen_ids[doc.doc_id] = st

        # ── Metadata completeness
        missing = [k for k in REQUIRED_METADATA if not doc.metadata.get(k)]
        if missing:
            r.metadata_issues.append({"id": doc.doc_id, "source_type": st,
                                       "missing": ", ".join(missing)})

        # ── PDF image analysis
        if st == "pdf":
            pdf_img_total += 1
            if doc.metadata.get("has_images"):
                pdf_img_with += 1
            strategy = doc.metadata.get("image_strategy", "none")
            ocr_hits  = doc.content.count("[Image text]:")
            cap_hits  = doc.content.count("[Figure]:")
            pdf_img_ocr_chars    += ocr_hits
            pdf_img_captioned    += cap_hits

        # ── Markdown — derived from content (not loader metadata)
        if st == "markdown":
            sigs = _derive_markdown_signals(doc.content)
            md_headings    += sigs["heading_count"]
            md_code_blocks += sigs["code_block_count"]

        # ── Code — derived from content
        if st == "code":
            sigs = _derive_code_signals(doc.content)
            code_funcs   += sigs["function_count"]
            code_classes += sigs["class_count"]
            if sigs["docstring_count"] == 0:
                code_no_docstring += 1

        # ── Web boilerplate
        if st == "web":
            bp = doc.metadata.get("boilerplate_removed_pct")
            if bp is not None:
                bp_pcts.append(bp)

        # ── Chunk readiness
        is_ready = (words >= 20 and not missing and nar < 0.05
                    and doc.doc_id not in r.duplicate_ids)
        if is_ready: ready += 1
        else:        not_ready += 1

        # ── Previews — keep first doc seen per source_type
        if st not in r.previews:
            r.previews[st] = {
                "doc_id":      doc.doc_id,
                "source_type": st,
                "file_name":   doc.metadata.get("file_name",
                               Path(doc.metadata.get("source","")).name),
                "words":       words,
                "has_images":  doc.metadata.get("has_images", False),
                "image_count": doc.metadata.get("image_count", 0),
                "image_strategy": doc.metadata.get("image_strategy", "n/a"),
                "has_tables":  doc.metadata.get("has_tables", False),
                "metadata_keys": [k for k in doc.metadata.keys()
                                  if not k.startswith("_")],
                "content_preview": doc.content[:400].replace("<","&lt;"),
                # Show where image data was injected (if any)
                "image_injections": (
                    [l for l in doc.content.splitlines()
                     if l.startswith("[Image text]:") or l.startswith("[Figure]:")]
                )[:3],
            }

    # Aggregate
    for st, wlist in source_words.items():
        r.docs_by_source[st]        = len(wlist)
        r.avg_words_by_source[st]   = round(sum(wlist)/len(wlist), 1)
        r.total_tokens_by_source[st] = source_tokens[st]

    r.projected_chunks_512 = r.total_estimated_tokens // 512 + 1
    r.chunk_readiness = {"ready": ready, "not_ready": not_ready}

    r.pdf_image_stats = {
        "total_pdf_pages":    pdf_img_total,
        "pages_with_images":  pdf_img_with,
        "strategy":           PDF_IMAGE_STRATEGY,
        "ocr_injections":     pdf_img_ocr_chars,
        "caption_injections": pdf_img_captioned,
    }
    total_md = source_words.get("markdown", [])
    r.markdown_stats = {
        "total_docs":     len(total_md),
        "total_headings": md_headings,
        "total_code_blocks": md_code_blocks,
    }
    total_code = source_words.get("code", [])
    r.code_stats = {
        "total_files":      len(total_code),
        "total_functions":  code_funcs,
        "total_classes":    code_classes,
        "no_docstring_pct": round(code_no_docstring / len(total_code) * 100)
                            if total_code else 0,
    }
    r.web_stats = {
        "total_pages": len(bp_pcts),
        "avg_boilerplate_pct": round(sum(bp_pcts)/len(bp_pcts), 1) if bp_pcts else 0,
    }
    return r


# ══════════════════════════════════════════════════════════
#  HTML REPORT GENERATOR
# ══════════════════════════════════════════════════════════

SRC_COLORS = {
    "pdf":      "#534AB7",
    "markdown": "#1D9E75",
    "code":     "#185FA5",
    "json_faq": "#BA7517",
    "csv":      "#993C1D",
    "web":      "#993556",
}

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:14px;
     background:#f8f7f4;color:#2c2c2a;line-height:1.6}
.page{max-width:980px;margin:0 auto;padding:24px 20px}
.rpt-hdr{background:#fff;border:0.5px solid #d3d1c7;border-radius:12px;
          padding:22px 28px;margin-bottom:14px}
.rpt-title{font-size:19px;font-weight:600;margin-bottom:3px}
.rpt-meta{font-size:12px;color:#888780}
.sec{background:#fff;border:0.5px solid #d3d1c7;border-radius:12px;
     padding:20px 24px;margin-bottom:12px}
.sec-title{font-size:15px;font-weight:600;margin-bottom:3px}
.sec-sub{font-size:12px;color:#888780;margin-bottom:14px}
.sec-note{font-size:12px;font-weight:500;color:#5f5e5a;margin-bottom:7px}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
.stat-card{background:#f8f7f4;border-radius:8px;padding:14px 16px}
.stat-label{font-size:11px;color:#888780;margin-bottom:3px;
            text-transform:uppercase;letter-spacing:.04em}
.stat-value{font-size:22px;font-weight:600}
.stat-sub{font-size:11px;color:#888780;margin-top:2px}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}
.chart{margin-bottom:8px}
.chart-title{font-size:11px;color:#888780;margin-bottom:8px;font-weight:500;
             text-transform:uppercase;letter-spacing:.04em}
.bar-row{display:flex;align-items:center;gap:8px;margin-bottom:5px}
.bar-label{font-size:12px;color:#5f5e5a;width:110px;flex-shrink:0;
           text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar-track{flex:1;background:#f1efe8;border-radius:4px;height:13px;overflow:hidden}
.bar-fill{height:100%;border-radius:4px}
.bar-val{font-size:12px;color:#2c2c2a;font-weight:500;min-width:50px;text-align:right}
table{width:100%;border-collapse:collapse;font-size:12px;margin-top:6px}
th{background:#f8f7f4;padding:7px 10px;text-align:left;font-weight:500;
   border-bottom:0.5px solid #d3d1c7;color:#5f5e5a}
td{padding:6px 10px;border-bottom:0.5px solid #f1efe8;
   max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ok-block{padding:9px 14px;border-radius:8px;font-size:13px;
          background:#EAF3DE;color:#3B6D11}
.warn-block{padding:9px 14px;border-radius:8px;font-size:13px;
            background:#FAEEDA;color:#633806}
.info-block{padding:9px 14px;border-radius:8px;font-size:13px;
            background:#E6F1FB;color:#0C447C;margin-bottom:8px}
.badge{font-size:11px;padding:2px 8px;border-radius:20px;font-weight:500}
.insight{font-size:12px;background:#f8f7f4;border-radius:6px;
         padding:10px 12px;color:#5f5e5a;margin-top:10px;line-height:1.6}
.findings{list-style:none;display:flex;flex-direction:column;gap:7px;margin-top:4px}
.findings li{font-size:13px;padding:10px 14px;background:#f8f7f4;
             border-radius:8px;line-height:1.5}
code{background:#f1efe8;padding:1px 5px;border-radius:4px;
     font-family:monospace;font-size:12px}
.note{font-size:11px;color:#888780;margin-top:5px}
/* ── Preview cards ── */
.preview-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.pcard{border:0.5px solid #d3d1c7;border-radius:10px;overflow:hidden}
.pcard-hdr{padding:10px 14px;display:flex;align-items:center;gap:8px}
.pcard-badge{font-size:11px;padding:3px 9px;border-radius:20px;font-weight:500}
.pcard-name{font-size:12px;color:#5f5e5a;flex:1;
            overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pcard-meta{padding:0 14px 6px;display:flex;flex-wrap:wrap;gap:5px}
.pcard-pill{font-size:11px;padding:2px 8px;border-radius:20px;
            background:#f1efe8;color:#5f5e5a}
.pcard-content{margin:0 14px 12px;padding:8px 10px;background:#f8f7f4;
               border-radius:6px;font-size:11px;font-family:monospace;
               color:#444441;white-space:pre-wrap;word-break:break-word;
               max-height:120px;overflow:hidden;line-height:1.5}
.pcard-img-inject{margin:0 14px 12px;padding:6px 10px;background:#EAF3DE;
                  border-radius:6px;font-size:11px;font-family:monospace;
                  color:#3B6D11;white-space:pre-wrap;word-break:break-word}
.pcard-meta-keys{margin:0 14px 10px;font-size:11px;color:#888780}
"""


def _sc(label, value, sub="", color="#1D9E75"):
    return (f'<div class="stat-card"><div class="stat-label">{label}</div>'
            f'<div class="stat-value" style="color:{color}">{value}</div>'
            + (f'<div class="stat-sub">{sub}</div>' if sub else "") + "</div>")


def _bar(data, title, color="#1D9E75"):
    if not data:
        return f"<p style='font-size:12px;color:#888'>No data</p>"
    mv = max(data.values()) or 1
    rows = "".join(
        f'<div class="bar-row">'
        f'<div class="bar-label">{l}</div>'
        f'<div class="bar-track"><div class="bar-fill" '
        f'style="width:{round(v/mv*100)}%;background:{color}"></div></div>'
        f'<div class="bar-val">{v:,}</div></div>'
        for l, v in sorted(data.items(), key=lambda x: -x[1])
    )
    return f'<div class="chart"><div class="chart-title">{title}</div>{rows}</div>'


def _issues(rows, cols, title, ok):
    if not rows:
        return f'<div class="ok-block">✓ {ok}</div>'
    h = "".join(f"<th>{c[1]}</th>" for c in cols)
    b = "".join(
        "<tr>" + "".join(f"<td>{r.get(c[0],'')}</td>" for c in cols) + "</tr>"
        for r in rows[:20]
    )
    more = f"<p class='note'>… and {len(rows)-20} more</p>" if len(rows) > 20 else ""
    return (f'<p class="sec-note">{title} — {len(rows)} issue(s)</p>'
            f'<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>{more}')


def _preview_card(p: Dict) -> str:
    st    = p["source_type"]
    color = SRC_COLORS.get(st, "#888780")
    bg    = color + "18"

    # Pills: words, images, tables, strategy
    pills = [f'<span class="pcard-pill">{p["words"]:,} words</span>']
    if p.get("has_images"):
        pills.append(
            f'<span class="pcard-pill" style="background:#534AB718;color:#534AB7">'
            f'📷 {p["image_count"]} image(s) — strategy: {p["image_strategy"]}</span>'
        )
    if p.get("has_tables"):
        pills.append('<span class="pcard-pill" style="background:#1D9E7518;color:#0F6E56">has tables</span>')

    pills_html = "".join(pills)

    # Image injection lines (OCR / caption output visible in content)
    inject_html = ""
    if p.get("image_injections"):
        lines = "\n".join(p["image_injections"])
        inject_html = (f'<div class="pcard-img-inject">'
                       f'Image content injected into doc:\n{lines}</div>')

    meta_keys = ", ".join(p["metadata_keys"][:12])
    if len(p["metadata_keys"]) > 12:
        meta_keys += f" … +{len(p['metadata_keys'])-12} more"

    return (
        f'<div class="pcard">'
        f'<div class="pcard-hdr" style="background:{bg}">'
        f'<span class="pcard-badge" style="background:{color}20;color:{color}">{st}</span>'
        f'<span class="pcard-name">{p["file_name"]}</span>'
        f'<span style="font-size:11px;color:#888780;font-family:monospace">{p["doc_id"]}</span>'
        f'</div>'
        f'<div class="pcard-meta">{pills_html}</div>'
        f'<div class="pcard-content">{p["content_preview"]}</div>'
        f'{inject_html}'
        f'<div class="pcard-meta-keys">metadata keys: {meta_keys}</div>'
        f'</div>'
    )


def generate_report(docs: List[Document], r: AnalysisResults) -> str:

    # ── Charts
    src_count  = _bar(r.docs_by_source, "Documents per source type", "#534AB7")
    src_tokens = _bar(r.total_tokens_by_source, "Tokens per source type", "#1D9E75")
    avg_words  = _bar({k: int(v) for k,v in r.avg_words_by_source.items()},
                      "Avg words per doc by source", "#185FA5")
    timing     = _bar({k: int(v*1000) for k,v in r.timing_by_source.items()},
                      "Loader time (ms)", "#BA7517")

    # ── Issue tables
    short_t = _issues(r.short_docs,
        [("id","Doc ID"),("source_type","Source"),("words","Words"),("preview","Preview")],
        "Short docs (< 20 words) — likely extraction failures",
        "No short documents found")
    long_t  = _issues(r.long_docs,
        [("id","Doc ID"),("source_type","Source"),("words","Words"),("source","File")],
        "Long docs (> 2,000 words) — will be split by chunker",
        "No excessively long documents")
    meta_t  = _issues(r.metadata_issues,
        [("id","Doc ID"),("source_type","Source"),("missing","Missing fields")],
        "Metadata completeness failures",
        "All documents have complete required metadata")

    dup_block = (
        '<div class="ok-block">✓ No duplicate doc_ids detected</div>'
        if not r.duplicate_count else
        f'<div class="warn-block">⚠ {r.duplicate_count} duplicate(s) — '
        f'{", ".join(r.duplicate_ids[:5])}{"…" if len(r.duplicate_ids)>5 else ""}</div>'
    )

    # ── PDF image section
    ps         = r.pdf_image_stats
    strategy   = ps.get("strategy","none")
    strat_color = {"none":"#888780","ocr":"#BA7517","caption":"#534AB7"}.get(strategy,"#888780")
    img_finding = ""
    if ps.get("pages_with_images", 0) > 0 and strategy == "none":
        img_finding = (
            f'<div class="warn-block" style="margin-top:10px">⚠ '
            f'{ps["pages_with_images"]} pages contain images but '
            f'<code>pdf_image_strategy="none"</code> — image content is silently dropped. '
            f'Set <code>PDF_IMAGE_STRATEGY="ocr"</code> (scanned docs) or '
            f'<code>"caption"</code> (diagrams/figures) to include image content.</div>'
        )
    elif strategy == "ocr":
        img_finding = (
            f'<div class="ok-block" style="margin-top:10px">✓ OCR strategy active — '
            f'{ps.get("ocr_injections",0)} [Image text]: block(s) injected into content.</div>'
        )
    elif strategy == "caption":
        img_finding = (
            f'<div class="ok-block" style="margin-top:10px">✓ Caption strategy active — '
            f'{ps.get("caption_injections",0)} [Figure]: block(s) injected into content.</div>'
        )

    # ── Source-specific insights
    md  = r.markdown_stats
    cd  = r.code_stats
    web = r.web_stats

    # ── Readiness
    ready     = r.chunk_readiness.get("ready", 0)
    not_ready = r.chunk_readiness.get("not_ready", 0)
    total_cr  = ready + not_ready
    rp        = round(ready / total_cr * 100) if total_cr else 0
    rc        = "#1D9E75" if rp>=80 else "#BA7517" if rp>=60 else "#E24B4A"

    # ── Source table rows
    src_rows = "".join(
        f"<tr>"
        f"<td><span class='badge' style='background:{SRC_COLORS.get(st,'#888')+'20'};"
        f"color:{SRC_COLORS.get(st,'#888')}'>{st}</span></td>"
        f"<td>{r.docs_by_source[st]:,}</td>"
        f"<td>{int(r.avg_words_by_source.get(st,0)):,}</td>"
        f"<td>{r.total_tokens_by_source.get(st,0):,}</td>"
        f"<td>{r.timing_by_source.get(st,0):.2f}s</td>"
        f"</tr>"
        for st in sorted(r.docs_by_source)
    )

    # ── Preview cards
    preview_cards = "".join(
        _preview_card(p)
        for st, p in sorted(r.previews.items())
    )

    # ── Actionable findings
    findings = []
    if r.short_docs:
        findings.append(f"🔴 <strong>{len(r.short_docs)} short doc(s)</strong> — extraction failures. Add <code>min_word_count=20</code> guard in Stage 2.")
    if r.duplicate_count:
        findings.append(f"🟡 <strong>{r.duplicate_count} duplicate(s)</strong> — <code>IngestionPipeline(dedup=True)</code> active but sources overlap.")
    if r.metadata_issues:
        findings.append(f"🔴 <strong>{len(r.metadata_issues)} metadata issue(s)</strong> — missing required fields will silently break retrieval filters. Fix loaders.")
    if ps.get("pages_with_images",0) > 0 and strategy == "none":
        findings.append(f"🟡 <strong>{ps['pages_with_images']} PDF pages</strong> have images but <code>strategy=none</code>. Switch to <code>ocr</code> or <code>caption</code> to include image content.")
    if r.long_docs:
        findings.append(f"🟡 <strong>{len(r.long_docs)} long doc(s)</strong> > 2,000 words. Use recursive chunker with <code>chunk_size=512, chunk_overlap=50</code>.")
    if cd.get("no_docstring_pct",0) > 50:
        findings.append(f"🟡 <strong>{cd['no_docstring_pct']}% of code files</strong> have no docstrings — poor embedding quality for code search. Add a summarization step in Stage 2.")
    if r.total_estimated_tokens > 500_000:
        cost = r.total_estimated_tokens / 1_000_000 * 0.13
        findings.append(f"💡 <strong>~{r.total_estimated_tokens:,} tokens</strong>. Embedding cost ~<strong>${cost:.3f}</strong> (text-embedding-3-small). Use batch embedding to stay within rate limits.")
    if rp == 100:
        findings.append("✅ All documents passed the chunk-readiness check. Corpus is clean — proceed to Stage 2.")
    if not findings:
        findings.append("✅ No critical issues detected.")
    findings_html = "".join(f"<li>{f}</li>" for f in findings)

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG Stage 1 — Ingestion Analysis</title>
<style>{CSS}</style></head><body><div class="page">

<div class="rpt-hdr">
  <div class="rpt-title">RAG Stage 1 — Ingestion Analysis Report</div>
  <div class="rpt-meta">
    {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} ·
    {r.total_docs} docs · {len(r.docs_by_source)} source types ·
    PDF image strategy: <code>{strategy}</code> ·
    pipeline: <code>rag_stage1_ingestion_pipeline.py</code>
  </div>
</div>

<div class="sec">
  <div class="sec-title">Pipeline Summary</div>
  <div class="sec-sub">High-level metrics across all ingested sources</div>
  <div class="stat-grid">
    {_sc("Total documents",   f"{r.total_docs:,}")}
    {_sc("Estimated tokens",  f"{r.total_estimated_tokens:,}", "~4 chars/token", "#534AB7")}
    {_sc("Projected chunks",  f"{r.projected_chunks_512:,}",   "at 512-token size", "#185FA5")}
    {_sc("Duplicates",        str(r.duplicate_count), "doc_id collisions",
         "#E24B4A" if r.duplicate_count else "#1D9E75")}
    {_sc("Metadata issues",   str(len(r.metadata_issues)), "missing required fields",
         "#E24B4A" if r.metadata_issues else "#1D9E75")}
    {_sc("Chunk-ready",       f"{rp}%", f"{ready}/{total_cr} docs pass all checks", rc)}
  </div>
</div>

<div class="sec">
  <div class="sec-title">Document Previews — one sample per source type</div>
  <div class="sec-sub">
    Shows exactly what each loader produces: content, metadata keys, word count,
    image detection, and — when strategy is ocr/caption — the injected image text.
  </div>
  <div class="preview-grid">{preview_cards}</div>
</div>

<div class="sec">
  <div class="sec-title">1 · Source Type Breakdown</div>
  <div class="sec-sub">Imbalances here cause retrieval bias — overrepresented sources dominate top-k results.</div>
  <table>
    <thead><tr><th>Source</th><th>Docs</th><th>Avg words</th>
    <th>Total tokens</th><th>Load time</th></tr></thead>
    <tbody>{src_rows}</tbody>
  </table>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:16px">
    {src_count}{avg_words}
  </div>
</div>

<div class="sec">
  <div class="sec-title">2 · Token Budget & Embedding Cost</div>
  <div class="sec-sub">Use this before choosing an embedding model tier or vector DB plan.</div>
  {src_tokens}
  <div class="stat-grid" style="margin-top:12px">
    {_sc("Total tokens",         f"{r.total_estimated_tokens:,}")}
    {_sc("Chunks @ 512t",        f"{r.projected_chunks_512:,}", "vectors in index")}
    {_sc("embed-3-small cost",   f"~${r.total_estimated_tokens/1_000_000*0.13:.4f}",
         "$0.13/1M tokens", "#BA7517")}
    {_sc("embed-3-large cost",   f"~${r.total_estimated_tokens/1_000_000*1.30:.4f}",
         "$1.30/1M tokens", "#993C1D")}
  </div>
</div>

<div class="sec">
  <div class="sec-title">3 · PDF Image Handling</div>
  <div class="sec-sub">
    Images are extracted via <code>fitz</code> (PyMuPDF). Strategy controls what
    happens to them: <code>none</code> → flagged only,
    <code>ocr</code> → Tesseract text appended,
    <code>caption</code> → Claude vision description appended.
  </div>
  <div class="stat-grid">
    {_sc("PDF pages parsed",     str(ps.get("total_pdf_pages",0)))}
    {_sc("Pages with images",    str(ps.get("pages_with_images",0)),
         "detected by fitz",
         "#534AB7" if ps.get("pages_with_images",0) else "#888780")}
    {_sc("Active strategy",      strategy, "none | ocr | caption", strat_color)}
    {_sc("OCR injections",       str(ps.get("ocr_injections",0)),
         "[Image text]: blocks added")}
    {_sc("Caption injections",   str(ps.get("caption_injections",0)),
         "[Figure]: blocks added")}
  </div>
  {img_finding}
  <div class="insight">
    💡 <strong>How to switch strategy:</strong> Change <code>PDF_IMAGE_STRATEGY</code>
    at the top of <code>analyze_ingestion.py</code>, or pass
    <code>pdf_image_strategy="ocr"</code> / <code>"caption"</code> to
    <code>DirectoryLoader</code> in production. The preview cards above show
    exactly what gets injected into each document's content when a strategy is active.
  </div>
</div>

<div class="sec">
  <div class="sec-title">4 · Loader Performance</div>
  <div class="sec-sub">PDF is CPU-bound — always uses <code>asyncio.to_thread()</code>. Web bottlenecks on network.</div>
  {timing}
</div>

<div class="sec">
  <div class="sec-title">5 · Short Document Audit</div>
  <div class="sec-sub">Under 20 words — likely extraction failures. Produce near-random embeddings.</div>
  {short_t}
</div>

<div class="sec">
  <div class="sec-title">6 · Long Document Audit</div>
  <div class="sec-sub">Over 2,000 words — Stage 2 chunker must split these. Ensure structure-aware splitting.</div>
  {long_t}
</div>

<div class="sec">
  <div class="sec-title">7 · Deduplication</div>
  <div class="sec-sub">Duplicates waste embedding compute and bias retrieval. Pipeline uses sha256 doc_ids.</div>
  {dup_block}
</div>

<div class="sec">
  <div class="sec-title">8 · Metadata Completeness</div>
  <div class="sec-sub">Required: <code>source</code>, <code>source_type</code>, <code>file_type</code>, <code>ingested_at</code>.</div>
  {meta_t}
</div>

<div class="sec">
  <div class="sec-title">9 · Source-specific Insights  <span style="font-size:12px;font-weight:400;color:#888780">(derived from content at analysis time, not baked into loaders)</span></div>
  <div class="sec-sub">Signals computed from doc.content inside run_analysis() — no extra metadata stored by loaders.</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px">
    <div class="stat-card">
      <div class="stat-label">Markdown</div>
      <div style="font-size:12px;color:#2c2c2a;margin-top:6px;line-height:2">
        {md.get("total_docs",0)} files<br>
        {md.get("total_headings",0)} headings<br>
        {md.get("total_code_blocks",0)} code blocks
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Code (Python)</div>
      <div style="font-size:12px;color:#2c2c2a;margin-top:6px;line-height:2">
        {cd.get("total_files",0)} files<br>
        {cd.get("total_functions",0)} functions<br>
        {cd.get("no_docstring_pct",0)}% files have no docstrings
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Web (HTML)</div>
      <div style="font-size:12px;color:#2c2c2a;margin-top:6px;line-height:2">
        {web.get("total_pages",0)} pages<br>
        avg {web.get("avg_boilerplate_pct",0)}% boilerplate removed
      </div>
    </div>
  </div>
</div>

<div class="sec">
  <div class="sec-title">10 · Chunk Readiness</div>
  <div class="sec-sub">words ≥ 20 + complete metadata + non-ASCII &lt; 5% + not a duplicate.</div>
  <div class="stat-grid">
    {_sc("Ready for Stage 2", str(ready),     f"{rp}% of corpus", "#1D9E75")}
    {_sc("Needs attention",   str(not_ready), "fails ≥ 1 check",
         "#E24B4A" if not_ready else "#888780")}
  </div>
</div>

<div class="sec">
  <div class="sec-title">Actionable Findings for Stage 2</div>
  <div class="sec-sub">Fix these before moving to Document Preprocessing.</div>
  <ul class="findings">{findings_html}</ul>
</div>

</div></body></html>"""


# ══════════════════════════════════════════════════════════
#  MAIN
#  Loader wiring:
#    DirectoryLoader → PDF, markdown (.md), code (.py)
#    JSONFAQLoader   → structured FAQ corpus
#    CSVProductLoader→ product catalog CSV
#    WebCacheLoader  → pre-fetched HTML articles
# ══════════════════════════════════════════════════════════

async def main():
    print("\n🔬 RAG Stage 1 — Ingestion Analysis")
    print(f"   PDF image strategy : {PDF_IMAGE_STRATEGY}")
    print(f"   Pipeline module    : rag_stage1_ingestion_pipeline.py")
    print("─" * 54)

    if not ASSETS.exists():
        print("❌  ./assets not found. Run: python setup_data.py first")
        return

    loaders_with_names: List[Tuple[str, BaseLoader]] = []

    # ── DirectoryLoader replaces all individual file-type loaders.
    #    pdf_image_strategy is forwarded through to PDFLoader for every .pdf found.
    if (ASSETS / "pdfs").exists():
        loaders_with_names.append((
            "pdf",
            DirectoryLoader(
                str(ASSETS / "pdfs"),
                glob_pattern="*.pdf",
                pdf_image_strategy=PDF_IMAGE_STRATEGY,
            ),
        ))

    if (ASSETS / "markdown").exists():
        loaders_with_names.append((
            "markdown",
            DirectoryLoader(str(ASSETS / "markdown"), glob_pattern="*.md"),
        ))

    if (ASSETS / "code").exists():
        loaders_with_names.append((
            "code",
            DirectoryLoader(str(ASSETS / "code"), glob_pattern="*.py"),
        ))

    # ── Custom loaders for sources DirectoryLoader cannot handle
    if (ASSETS / "json" / "faq_corpus.json").exists():
        loaders_with_names.append((
            "json_faq",
            JSONFAQLoader(ASSETS / "json" / "faq_corpus.json"),
        ))

    if (ASSETS / "csv" / "product_catalog.csv").exists():
        loaders_with_names.append((
            "csv",
            CSVProductLoader(
                ASSETS / "csv" / "product_catalog.csv",
                content_columns=["name", "category", "description", "tags"],
                metadata_columns=["id", "category", "price_usd", "in_stock", "rating"],
            ),
        ))

    if (ASSETS / "web_cache").exists():
        loaders_with_names.append((
            "web",
            WebCacheLoader(ASSETS / "web_cache"),
        ))

    # ── Run each loader, record timing per source
    all_docs: List[Document] = []
    timings:  Dict[str, float] = {}

    for name, loader in loaders_with_names:
        print(f"\n📂 {name.upper()}")
        t0   = time.monotonic()
        docs = await loader.load_safe()
        timings[name] = round(time.monotonic() - t0, 2)
        print(f"   → {len(docs)} docs in {timings[name]}s")
        all_docs.extend(docs)

    # ── Dedup + min-word filter via IngestionPipeline logic
    print(f"\n{'─'*54}")
    print("Applying dedup + min-word filter (IngestionPipeline)...")
    seen:       set = set()
    final_docs: List[Document] = []
    for doc in all_docs:
        if doc.is_empty() or doc.word_count() < 5:
            continue
        if doc.doc_id in seen:
            continue
        seen.add(doc.doc_id)
        final_docs.append(doc)

    print(f"  {len(all_docs)} raw → {len(final_docs)} after dedup & filter")

    # ── Analysis
    print("\nComputing analysis metrics...")
    results = run_analysis(final_docs, timings)

    print(f"""
  Summary
  ──────────────────────────────────────
  Total docs       : {results.total_docs:,}
  Tokens (est.)    : {results.total_estimated_tokens:,}
  Chunks @ 512t    : {results.projected_chunks_512:,}
  Duplicates       : {results.duplicate_count}
  Metadata issues  : {len(results.metadata_issues)}
  Short docs       : {len(results.short_docs)}
  Long docs        : {len(results.long_docs)}
  PDF pages w/ img : {results.pdf_image_stats.get('pages_with_images',0)}
  Image strategy   : {results.pdf_image_stats.get('strategy','none')}
  Chunk-ready      : {results.chunk_readiness.get('ready',0)}/{results.total_docs}
    """)

    report = generate_report(final_docs, results)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"✅  Report → {REPORT_PATH.resolve()}")
    print(f"    Open:   open {REPORT_PATH}\n")


if __name__ == "__main__":
    asyncio.run(main())