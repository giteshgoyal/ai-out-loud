"""
RAG Pipeline — Stage 1: Data Sources & Ingestion
Production-grade, reusable ingestion framework.

Architecture decisions baked in:
  - Async-first: all loaders run concurrently, CPU-bound work offloaded to threads
  - Error-isolated: one failing loader never kills the pipeline
  - Deterministic IDs: sha256 of source+content → safe deduplication across runs
  - Normalized schema: ALL sources produce the same Document shape
  - Delta-ready: IncrementalTracker enables re-run without full re-ingestion

Install:
  pip install pymupdf4llm httpx beautifulsoup4 asyncpg

Usage:
  See build_knowledge_base_pipeline() at the bottom for a complete example.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
#  CORE DATA MODEL
#  The single contract between Stage 1 and everything downstream.
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Document:
    """
    Universal document schema.
    Every loader produces Documents. No stage downstream ever receives
    anything else. This is what makes the pipeline composable.

    doc_id:   Deterministic sha256 of (source + content).
              Same document ingested twice → same ID → safe to dedup.
              For API sources, override with the external record ID
              to track updates across runs.

    metadata: Open dict, but always includes the base keys injected by
              build_metadata(). Downstream filters and eval pipelines
              depend on those keys being present.
    """
    content: str
    metadata: Dict[str, Any]
    doc_id: str = field(default="")

    def __post_init__(self):
        if not self.doc_id:
            fingerprint = f"{self.metadata.get('source', '')}{self.content}"
            self.doc_id = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]

    def is_empty(self) -> bool:
        return not self.content.strip()

    def word_count(self) -> int:
        return len(self.content.split())

    def char_count(self) -> int:
        return len(self.content)

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return (
            f"Document(id={self.doc_id}, words={self.word_count()}, "
            f"source={self.metadata.get('source_type', '?')}, "
            f"preview='{preview}...')"
        )


def build_metadata(source: str, source_type: str, file_type: str, **extra) -> Dict[str, Any]:
    """
    Enforces consistent metadata schema across all loaders.
    ALWAYS call this — never build raw dicts in loaders.

    Required keys that every downstream stage relies on:
      source       — where the document came from (path, URL, DSN host)
      source_type  — category: pdf | web | sql | api | code | file
      file_type    — format:   pdf | html | json | csv | py | md | ...
      ingested_at  — ISO timestamp of when this document was created
    """
    return {
        "source":      source,
        "source_type": source_type,
        "file_type":   file_type,
        "ingested_at": datetime.utcnow().isoformat(),
        **extra,
    }


# ══════════════════════════════════════════════════════════════════════
#  BASE LOADER ABSTRACTION
#  The only interface the pipeline orchestrator cares about.
# ══════════════════════════════════════════════════════════════════════

class BaseLoader(ABC):
    """
    Implement load() in every subclass.
    Call load_safe() from the pipeline — it wraps load() with an error
    boundary so a broken loader never crashes the whole pipeline.
    """

    @abstractmethod
    async def load(self) -> List[Document]:
        ...

    async def load_safe(self) -> List[Document]:
        try:
            docs = await self.load()
            docs = [d for d in docs if not d.is_empty()]
            logger.info(f"[{self.__class__.__name__}] loaded {len(docs)} documents")
            return docs
        except Exception as exc:
            logger.error(f"[{self.__class__.__name__}] failed: {exc}", exc_info=True)
            return []


# ══════════════════════════════════════════════════════════════════════
#  LOADER: PDF
#  Uses pymupdf4llm for layout-aware extraction.
#  Outputs markdown-formatted text including tables.
# ══════════════════════════════════════════════════════════════════════

class PDFLoader(BaseLoader):
    """
    Layout-aware PDF loading with pymupdf4llm + configurable image handling.

    image_strategy controls what happens to images on each page:
      "none"    — detect & flag images (has_images metadata), content dropped. Default.
      "ocr"     — run Tesseract OCR on each image, append text to page content.
                  Best for: scanned docs, forms, stamps, receipts.
                  Requires: pip install pytesseract pillow  + Tesseract binary
      "caption" — send each image to a vision LLM, append description to content.
                  Best for: diagrams, architecture figures, charts.
                  Requires: ANTHROPIC_API_KEY env var  + pip install httpx

    Image bytes are extracted via fitz (PyMuPDF) — the same library pymupdf4llm
    wraps — so no extra install is needed beyond pymupdf4llm itself.
    """

    def __init__(
        self,
        file_path: str,
        min_chars_per_page: int = 50,
        image_strategy: str = "none",       # "none" | "ocr" | "caption"
        vision_api_key: Optional[str] = None,
    ):
        self.file_path        = Path(file_path)
        self.min_chars_per_page = min_chars_per_page
        self.image_strategy   = image_strategy
        self.vision_api_key   = vision_api_key

    # ── Public entry point ──────────────────────────────────────────────

    async def load(self) -> List[Document]:
        # CPU-bound markdown extraction — must run in thread, never in event loop
        docs = await asyncio.to_thread(self._load_sync)

        # Async captioning happens here (after sync parse), using gathered image bytes
        if self.image_strategy == "caption":
            docs = await self._run_captioning(docs)

        return docs

    # ── Sync parse (runs in thread pool) ────────────────────────────────

    def _load_sync(self) -> List[Document]:
        try:
            import pymupdf4llm
            import fitz                     # PyMuPDF — bundled with pymupdf4llm
        except ImportError:
            raise ImportError("pip install pymupdf4llm")

        pages     = pymupdf4llm.to_markdown(str(self.file_path), page_chunks=True)
        fitz_doc  = fitz.open(str(self.file_path))
        total_pgs = len(pages)
        documents = []

        for page_dict in pages:
            content  = page_dict.get("text", "").strip()
            page_meta = page_dict.get("metadata", {})
            page_num  = page_meta.get("page", 0) + 1       # 0-indexed → 1-indexed

            # Extract image bytes from this page (needed for ocr + caption)
            img_bytes_list = self._extract_page_images(fitz_doc, page_num - 1)
            has_images     = bool(img_bytes_list)

            # Strategy: OCR — runs synchronously here in the thread
            if self.image_strategy == "ocr" and has_images:
                ocr_parts = []
                for img_bytes in img_bytes_list:
                    ocr_text = self._ocr_image(img_bytes)
                    if ocr_text:
                        ocr_parts.append(f"[Image text]: {ocr_text}")
                if ocr_parts:
                    content = content + "\n" + "\n".join(ocr_parts)

            if len(content) < self.min_chars_per_page:
                continue    # skip blank pages / dividers even after OCR

            doc = Document(
                content=content,
                metadata=build_metadata(
                    source=str(self.file_path),
                    source_type="pdf",
                    file_type="pdf",
                    file_name=self.file_path.name,
                    page_number=page_num,
                    total_pages=total_pgs,
                    has_tables="| " in content,
                    has_images=has_images,
                    image_count=len(img_bytes_list),
                    image_strategy=self.image_strategy,
                ),
            )

            # Strategy: caption — stash raw bytes for async processing in load()
            # Stored under a private key; removed after captioning is done.
            if self.image_strategy == "caption" and has_images:
                doc.metadata["_image_bytes"] = img_bytes_list

            documents.append(doc)

        fitz_doc.close()
        return documents

    # ── Image extraction (fitz) ─────────────────────────────────────────

    @staticmethod
    def _extract_page_images(fitz_doc, page_index: int) -> List[bytes]:
        """Return raw image bytes for every image on the given page."""
        try:
            page       = fitz_doc[page_index]
            img_list   = page.get_images(full=True)
            result     = []
            for img_info in img_list:
                xref       = img_info[0]
                base_image = fitz_doc.extract_image(xref)
                if base_image and base_image.get("image"):
                    result.append(base_image["image"])
            return result
        except Exception as exc:
            logger.warning(f"Image extraction failed on page {page_index}: {exc}")
            return []

    # ── OCR strategy (sync, runs inside thread) ─────────────────────────

    @staticmethod
    def _ocr_image(img_bytes: bytes) -> str:
        """Run Tesseract OCR on raw image bytes. Returns extracted text."""
        try:
            from PIL import Image
            import pytesseract, io
            img = Image.open(io.BytesIO(img_bytes))
            return pytesseract.image_to_string(img).strip()
        except ImportError:
            raise ImportError("pip install pytesseract pillow  (and install Tesseract binary)")
        except Exception as exc:
            logger.warning(f"OCR failed: {exc}")
            return ""

    # ── Caption strategy (async, runs after sync parse) ─────────────────

    async def _run_captioning(self, docs: List[Document]) -> List[Document]:
        """
        For each Document that has _image_bytes in metadata, send each image
        to Claude claude-haiku-4-5-20251001 and append the captions to content.
        Removes the private _image_bytes key after processing.
        """
        import os
        try:
            import httpx, base64
        except ImportError:
            raise ImportError("pip install httpx")

        api_key = self.vision_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.warning(
                "image_strategy='caption' requires ANTHROPIC_API_KEY env var. "
                "Skipping captioning — images will not be described."
            )
            for doc in docs:
                doc.metadata.pop("_image_bytes", None)
            return docs

        async with httpx.AsyncClient(timeout=30) as client:
            for doc in docs:
                img_bytes_list = doc.metadata.pop("_image_bytes", [])
                if not img_bytes_list:
                    continue

                captions = []
                for img_bytes in img_bytes_list:
                    b64 = base64.b64encode(img_bytes).decode()
                    try:
                        resp = await client.post(
                            "https://api.anthropic.com/v1/messages",
                            headers={
                                "x-api-key":          api_key,
                                "anthropic-version":  "2023-06-01",
                                "content-type":       "application/json",
                            },
                            json={
                                "model":      "claude-haiku-4-5-20251001",
                                "max_tokens": 150,
                                "messages": [{
                                    "role": "user",
                                    "content": [
                                        {"type": "image",
                                         "source": {"type": "base64",
                                                    "media_type": "image/png",
                                                    "data": b64}},
                                        {"type": "text",
                                         "text": (
                                             "Describe this image in 2 sentences for a "
                                             "search index. Focus on what it shows "
                                             "(data, structure, concepts) not how it looks."
                                         )},
                                    ],
                                }],
                            },
                        )
                        resp.raise_for_status()
                        caption = resp.json()["content"][0]["text"].strip()
                        captions.append(caption)
                    except Exception as exc:
                        logger.warning(f"Caption API call failed: {exc}")

                if captions:
                    doc.content += "\n" + "\n".join(f"[Figure]: {c}" for c in captions)
                    doc.metadata["image_captions"] = captions

        return docs


# ══════════════════════════════════════════════════════════════════════
#  LOADER: WEB SCRAPER
#  Async bulk loader with boilerplate stripping and rate limiting.
# ══════════════════════════════════════════════════════════════════════

class WebLoader(BaseLoader):
    """
    Fetches and cleans a list of URLs concurrently.

    Strips nav/footer/script/style/aside before creating the Document
    so chunking and embedding see only the actual content.

    rate_limit_per_second: courtesy delay between requests.
    Set to 0 for internal services you control.
    """

    def __init__(
        self,
        urls: List[str],
        rate_limit_per_second: float = 2.0,
        timeout: int = 30,
        user_agent: str = "RAG-ingestion-bot/1.0",
    ):
        self.urls = urls
        self.delay = 1.0 / rate_limit_per_second if rate_limit_per_second > 0 else 0
        self.timeout = timeout
        self.user_agent = user_agent

    async def load(self) -> List[Document]:
        try:
            import httpx
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("pip install httpx beautifulsoup4")

        documents = []

        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            headers={"User-Agent": self.user_agent},
        ) as client:
            for url in self.urls:
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()

                    soup = BeautifulSoup(resp.text, "html.parser")

                    # Strip boilerplate — do this BEFORE get_text()
                    for tag in soup.find_all(
                        ["nav", "footer", "script", "style", "aside", "header",
                         "noscript", "iframe", "form"]
                    ):
                        tag.decompose()

                    text = soup.get_text(separator="\n", strip=True)
                    title_tag = soup.find("title")

                    # Skip pages with near-no content (login walls, 403 pages, etc.)
                    if len(text.split()) < 20:
                        logger.warning(f"Skipping near-empty page: {url}")
                        continue

                    documents.append(Document(
                        content=text,
                        metadata=build_metadata(
                            source=url,
                            source_type="web",
                            file_type="html",
                            title=title_tag.string.strip() if title_tag else "",
                            status_code=resp.status_code,
                            content_type=resp.headers.get("content-type", ""),
                        ),
                    ))

                    if self.delay > 0:
                        await asyncio.sleep(self.delay)

                except Exception as exc:
                    logger.warning(f"Failed to load {url}: {exc}")

        return documents


# ══════════════════════════════════════════════════════════════════════
#  LOADER: SQL DATABASE
#  Turns database rows into Documents.
#  Each row = one document. Content built from specified columns.
# ══════════════════════════════════════════════════════════════════════

class SQLLoader(BaseLoader):
    """
    Load rows from a SQL database as Documents.

    content_columns:  columns whose values form the document body.
                      Joined as "col: value\n" to preserve column semantics.
    metadata_columns: columns to include as metadata for filtering.

    Example — loading a product FAQ table:
        SQLLoader(
            dsn="postgresql://user:pass@host/db",
            query="SELECT id, question, answer, category FROM faq WHERE active=true",
            content_columns=["question", "answer"],
            metadata_columns=["id", "category"],
        )
    """

    def __init__(
        self,
        dsn: str,
        query: str,
        content_columns: List[str],
        metadata_columns: Optional[List[str]] = None,
    ):
        self.dsn = dsn
        self.query = query
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns or []

    async def load(self) -> List[Document]:
        try:
            import asyncpg
        except ImportError:
            raise ImportError("pip install asyncpg")

        conn = await asyncpg.connect(self.dsn)
        try:
            rows = await conn.fetch(self.query)
        finally:
            await conn.close()

        # Sanitize DSN for metadata — never store credentials
        safe_source = self.dsn.split("@")[-1] if "@" in self.dsn else self.dsn

        documents = []
        for row in rows:
            row_dict = dict(row)

            # Build content preserving column semantics
            content_parts = [
                f"{col}: {row_dict[col]}"
                for col in self.content_columns
                if row_dict.get(col) is not None
            ]
            content = "\n".join(content_parts)

            extra_meta = {col: str(row_dict.get(col, "")) for col in self.metadata_columns}

            documents.append(Document(
                content=content,
                metadata=build_metadata(
                    source=safe_source,
                    source_type="database",
                    file_type="sql",
                    **extra_meta,
                ),
            ))

        return documents


# ══════════════════════════════════════════════════════════════════════
#  LOADER: DIRECTORY (multi-format file walker)
#  Recurses a directory and routes each file to the right loader.
# ══════════════════════════════════════════════════════════════════════

class DirectoryLoader(BaseLoader):
    """
    Recursively loads all supported files in a directory.
    Routes each file to the appropriate loader by extension.

    Add new extensions to EXTENSION_HANDLERS to extend coverage.
    """

    # Mapping from extension → (source_type, file_type)
    TEXT_EXTENSIONS: Dict[str, str] = {
        ".md":   "markdown",
        ".txt":  "text",
        ".py":   "code",
        ".js":   "code",
        ".ts":   "code",
        ".go":   "code",
        ".java": "code",
        ".yaml": "config",
        ".yml":  "config",
        ".json": "config",
        ".toml": "config",
    }

    def __init__(
        self,
        directory: str,
        glob_pattern: str = "**/*",
        include_hidden: bool = False,
        min_chars: int = 100,
        pdf_image_strategy: str = "none",   # passed through to PDFLoader for every PDF found
        pdf_vision_api_key: Optional[str] = None,
    ):
        self.directory          = Path(directory)
        self.glob_pattern       = glob_pattern
        self.include_hidden     = include_hidden
        self.min_chars          = min_chars
        self.pdf_image_strategy = pdf_image_strategy
        self.pdf_vision_api_key = pdf_vision_api_key

    async def load(self) -> List[Document]:
        all_files = [
            f for f in self.directory.glob(self.glob_pattern)
            if f.is_file() and (self.include_hidden or not f.name.startswith("."))
        ]

        # Batch into parallel tasks — PDFs go async-to-thread, text files are fast
        tasks = [self._load_file(f) for f in all_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        docs = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"File load failed: {result}")
            else:
                docs.extend(result)

        return docs

    async def _load_file(self, path: Path) -> List[Document]:
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            loader = PDFLoader(
                str(path),
                image_strategy=self.pdf_image_strategy,
                vision_api_key=self.pdf_vision_api_key,
            )
            return await loader.load_safe()

        if suffix in self.TEXT_EXTENSIONS:
            return await self._load_text(path, self.TEXT_EXTENSIONS[suffix])

        logger.debug(f"Skipping unsupported extension: {path}")
        return []

    async def _load_text(self, path: Path, source_type: str) -> List[Document]:
        try:
            content = await asyncio.to_thread(
                path.read_text, encoding="utf-8", errors="ignore"
            )
            if len(content) < self.min_chars:
                return []

            return [Document(
                content=content,
                metadata=build_metadata(
                    source=str(path),
                    source_type=source_type,
                    file_type=path.suffix.lstrip("."),
                    file_name=path.name,
                    file_size_bytes=path.stat().st_size,
                ),
            )]
        except Exception as exc:
            logger.warning(f"Could not read {path}: {exc}")
            return []


# ══════════════════════════════════════════════════════════════════════
#  LOADER: REST API (paginated)
#  Works with Notion, Confluence, Zendesk, any paginated JSON API.
# ══════════════════════════════════════════════════════════════════════

class APILoader(BaseLoader):
    """
    Fetches paginated JSON from a REST API endpoint.

    content_field:  key in each item that holds the text content.
    id_field:       key for the external record ID (used as doc_id base
                    so updates are tracked across ingestion runs).
    pagination_key: JSON key holding the URL of the next page (or None).

    For APIs with cursor-based pagination, override _get_next_url().
    """

    def __init__(
        self,
        url: str,
        headers: Dict[str, str],
        items_path: str,            # dot-separated path to the items array, e.g. "results"
        content_field: str = "body",
        id_field: str = "id",
        title_field: str = "title",
        pagination_key: str = "next",
        max_pages: int = 200,
    ):
        self.url = url
        self.headers = headers
        self.items_path = items_path.split(".")
        self.content_field = content_field
        self.id_field = id_field
        self.title_field = title_field
        self.pagination_key = pagination_key
        self.max_pages = max_pages

    async def load(self) -> List[Document]:
        try:
            import httpx
        except ImportError:
            raise ImportError("pip install httpx")

        documents = []
        next_url: Optional[str] = self.url
        page = 0

        async with httpx.AsyncClient(headers=self.headers, timeout=30) as client:
            while next_url and page < self.max_pages:
                resp = await client.get(next_url)
                resp.raise_for_status()
                data = resp.json()

                items = self._extract_items(data)
                for item in items:
                    content = str(item.get(self.content_field) or "").strip()
                    if not content:
                        continue

                    external_id = str(item.get(self.id_field, ""))
                    title = str(item.get(self.title_field, ""))

                    documents.append(Document(
                        content=content,
                        metadata=build_metadata(
                            source=self.url,
                            source_type="api",
                            file_type="json",
                            external_id=external_id,
                            title=title,
                        ),
                        # Use external ID as doc_id → stable across re-ingestion runs
                        doc_id=f"api_{external_id}" if external_id else "",
                    ))

                next_url = data.get(self.pagination_key)
                page += 1

        return documents

    def _extract_items(self, data: dict) -> List[dict]:
        result = data
        for key in self.items_path:
            result = result.get(key, {}) if isinstance(result, dict) else {}
        return result if isinstance(result, list) else []


# ══════════════════════════════════════════════════════════════════════
#  INGESTION PIPELINE ORCHESTRATOR
#  Runs all loaders concurrently, deduplicates, filters, reports stats.
# ══════════════════════════════════════════════════════════════════════

@dataclass
class IngestionStats:
    total_loaded:       int = 0
    total_after_dedup:  int = 0
    total_filtered:     int = 0   # empty or below min_word_count
    loaders_failed:     int = 0
    duration_seconds:   float = 0.0


class IngestionPipeline:
    """
    Orchestrates a list of loaders with:
      - Concurrent execution (all loaders run in parallel)
      - Per-loader error isolation (one failure doesn't stop others)
      - Deduplication by doc_id
      - Min word count filter
      - Structured stats for observability

    Usage:
        pipeline = IngestionPipeline(loaders=[...])
        docs, stats = await pipeline.run()
        print(f"Got {stats.total_after_dedup} unique docs in {stats.duration_seconds}s")
    """

    def __init__(
        self,
        loaders: List[BaseLoader],
        dedup: bool = True,
        min_word_count: int = 5,
    ):
        self.loaders = loaders
        self.dedup = dedup
        self.min_word_count = min_word_count

    async def run(self) -> tuple[List[Document], IngestionStats]:
        import time
        t0 = time.monotonic()

        # All loaders run concurrently
        raw_results = await asyncio.gather(
            *[loader.load_safe() for loader in self.loaders],
            return_exceptions=True,
        )

        stats = IngestionStats()
        seen_ids: set[str] = set()
        final_docs: List[Document] = []

        for result in raw_results:
            if isinstance(result, Exception):
                stats.loaders_failed += 1
                continue

            for doc in result:
                stats.total_loaded += 1

                if doc.word_count() < self.min_word_count:
                    stats.total_filtered += 1
                    continue

                if self.dedup:
                    if doc.doc_id in seen_ids:
                        stats.total_filtered += 1
                        continue
                    seen_ids.add(doc.doc_id)

                final_docs.append(doc)

        stats.total_after_dedup = len(final_docs)
        stats.duration_seconds = round(time.monotonic() - t0, 2)

        logger.info(
            f"Ingestion complete — "
            f"{stats.total_loaded} raw → {stats.total_after_dedup} unique "
            f"({stats.total_filtered} filtered) in {stats.duration_seconds}s"
        )

        return final_docs, stats


# ══════════════════════════════════════════════════════════════════════
#  DELTA / INCREMENTAL INGESTION
#  Tracks seen doc_ids so re-runs only process new/changed documents.
#  Swap the JSON file for Redis or a DB table in production.
# ══════════════════════════════════════════════════════════════════════

class IncrementalTracker:
    """
    Prevents re-processing documents on subsequent ingestion runs.

    How it works:
      1. Load previous state (set of processed doc_ids + timestamps)
      2. filter_new() removes docs already in state
      3. mark_processed() updates state after successful downstream processing
      4. State persists to disk (or Redis/DB in production)

    This pattern is safe for:
      - Periodic re-scrapes of web sources (new pages picked up, old skipped)
      - Nightly database exports (only new/updated rows processed)
      - Watched directories (only changed files processed)

    For API sources: pair with a query filter like "WHERE updated_at > last_run_ts"
    for efficiency, then use this tracker as a safety net for dedup.
    """

    def __init__(self, state_file: str = ".ingestion_state.json"):
        self.state_file = Path(state_file)
        self._state: Dict[str, str] = self._load()

    def _load(self) -> Dict[str, str]:
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return {}

    def _save(self):
        self.state_file.write_text(json.dumps(self._state, indent=2))

    def filter_new(self, docs: List[Document]) -> List[Document]:
        """Return only docs not previously processed."""
        new = [d for d in docs if d.doc_id not in self._state]
        skipped = len(docs) - len(new)
        if skipped:
            logger.info(f"IncrementalTracker: skipped {skipped} already-processed docs")
        return new

    def mark_processed(self, docs: List[Document]):
        """Call this AFTER docs have been successfully chunked and indexed."""
        for doc in docs:
            self._state[doc.doc_id] = doc.metadata.get("ingested_at", "")
        self._save()

    def reset(self):
        """Full re-ingestion on next run. Use when index is rebuilt from scratch."""
        self._state = {}
        self._save()

    @property
    def processed_count(self) -> int:
        return len(self._state)


# ══════════════════════════════════════════════════════════════════════
#  USAGE EXAMPLE — Enterprise Knowledge Base
#  Shows combining multiple source types into one pipeline run.
# ══════════════════════════════════════════════════════════════════════

async def build_knowledge_base_pipeline():
    """
    Example: enterprise knowledge base combining internal PDFs,
    product documentation website, code repository, and CRM FAQ table.
    """

    pipeline = IngestionPipeline(
        loaders=[
            # Internal documentation (PDFs + markdown runbooks)
            DirectoryLoader("./docs/internal", glob_pattern="**/*.pdf"),
            DirectoryLoader("./docs/runbooks", glob_pattern="**/*.md"),

            # Product documentation website
            WebLoader(
                urls=[
                    "https://docs.yourproduct.com/getting-started",
                    "https://docs.yourproduct.com/api-reference",
                    "https://docs.yourproduct.com/tutorials",
                ],
                rate_limit_per_second=2.0,
            ),

            # Codebase (for developer copilot use case)
            DirectoryLoader("./src", glob_pattern="**/*.py"),

            # CRM FAQ corpus
            SQLLoader(
                dsn="postgresql://user:pass@prod-db.internal/crm",
                query="""
                    SELECT id, title, question, answer, category, updated_at
                    FROM faq_articles
                    WHERE published = true
                      AND language = 'en'
                    ORDER BY updated_at DESC
                """,
                content_columns=["question", "answer"],
                metadata_columns=["id", "title", "category", "updated_at"],
            ),
        ],
        dedup=True,
        min_word_count=10,
    )

    docs, stats = await pipeline.run()

    print(f"""
    ✓ Ingestion complete
      Loaded:    {stats.total_loaded} raw documents
      Unique:    {stats.total_after_dedup} after deduplication
      Filtered:  {stats.total_filtered} (empty / too short)
      Failed:    {stats.loaders_failed} loaders
      Time:      {stats.duration_seconds}s
    """)

    # For incremental runs — only pass new docs to next stage
    tracker = IncrementalTracker(".ingestion_state.json")
    new_docs = tracker.filter_new(docs)

    print(f"  New/changed: {len(new_docs)} docs to process downstream\n")

    # After successful chunking + indexing downstream:
    # tracker.mark_processed(new_docs)

    return new_docs, stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(build_knowledge_base_pipeline())