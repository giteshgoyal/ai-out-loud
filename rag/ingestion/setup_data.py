"""
RAG Stage 1 — Test Data Setup
Downloads diverse, real-world data into an ./assets directory.
Run this once before running analyze_ingestion.py

Usage:
    pip install httpx
    python setup_data.py

What gets downloaded and WHY each type matters for RAG:
  assets/pdfs/       — layout-heavy documents: tests table extraction, multi-page handling
  assets/markdown/   — structured text: tests heading/code-block handling
  assets/code/       — Python source: tests structure-aware metadata enrichment
  assets/json/       — FAQ corpus: tests schema normalization, nested payloads
  assets/csv/        — tabular data: tests column-semantic preservation
  assets/web_cache/  — pre-fetched HTML: tests boilerplate stripping
"""

import asyncio
import json
import textwrap
from pathlib import Path

ASSETS = Path("./assets")


def make_dirs():
    for d in ["pdfs", "markdown", "code", "json", "csv", "web_cache"]:
        (ASSETS / d).mkdir(parents=True, exist_ok=True)
    print("✓ Created asset directories")


# ──────────────────────────────────────────────
# PDFs — real research papers with tables, math,
# citations, figures. The hardest format to parse.
# ──────────────────────────────────────────────

async def download_pdfs():
    try:
        import httpx
    except ImportError:
        print("  pip install httpx  — then re-run")
        return

    # Publicly available papers on arXiv
    pdfs = {
        "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762",
        "rag_original_paper.pdf":        "https://arxiv.org/pdf/2005.11401",
        "hyde_paper.pdf":                "https://arxiv.org/pdf/2212.10496",
        "sentence_bert.pdf":             "https://arxiv.org/pdf/1908.10084",
    }

    async with httpx.AsyncClient(timeout=60, follow_redirects=True,
            headers={"User-Agent": "RAG-test-setup/1.0"}) as client:
        for name, url in pdfs.items():
            dest = ASSETS / "pdfs" / name
            if dest.exists():
                print(f"  ↩  {name} already exists, skipping")
                continue
            try:
                print(f"  ↓  Downloading {name} ...")
                r = await client.get(url)
                r.raise_for_status()
                dest.write_bytes(r.content)
                print(f"  ✓  {name} ({len(r.content)//1024} KB)")
            except Exception as e:
                print(f"  ✗  {name}: {e}")

    print("✓ PDFs done\n")


# ──────────────────────────────────────────────
# Markdown — README files from major RAG-related
# open-source projects. Rich heading structure,
# code fences, tables, badges.
# ──────────────────────────────────────────────

async def download_markdown():
    try:
        import httpx
    except ImportError:
        return

    md_files = {
        "langchain_readme.md":   "https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md",
        "llama_index_readme.md": "https://raw.githubusercontent.com/run-llama/llama_index/main/README.md",
        "qdrant_readme.md":      "https://raw.githubusercontent.com/qdrant/qdrant/master/README.md",
        "chroma_readme.md":      "https://raw.githubusercontent.com/chroma-core/chroma/main/README.md",
        "ragas_readme.md":       "https://raw.githubusercontent.com/explodinggradients/ragas/main/README.md",
    }

    async with httpx.AsyncClient(timeout=30, follow_redirects=True,
            headers={"User-Agent": "RAG-test-setup/1.0"}) as client:
        for name, url in md_files.items():
            dest = ASSETS / "markdown" / name
            if dest.exists():
                print(f"  ↩  {name} already exists, skipping")
                continue
            try:
                r = await client.get(url)
                r.raise_for_status()
                dest.write_bytes(r.content)
                print(f"  ✓  {name} ({len(r.content)//1024} KB)")
            except Exception as e:
                print(f"  ✗  {name}: {e}")

    print("✓ Markdown done\n")


# ──────────────────────────────────────────────
# Python source code — CPython stdlib modules.
# Mix of: dense docstrings, pure logic,
# small helpers, and large complex modules.
# ──────────────────────────────────────────────

async def download_code():
    try:
        import httpx
    except ImportError:
        return

    base = "https://raw.githubusercontent.com/python/cpython/main/Lib/"
    modules = [
        "pathlib.py",       # large, well-documented
        "asyncio/__init__.py",
        "json/__init__.py",
        "logging/__init__.py",
        "textwrap.py",      # small, simple
        "urllib/parse.py",
        "dataclasses.py",   # heavy docstrings
        "functools.py",
        "itertools.pyi",    # type stubs — different pattern
        "hashlib.py",
    ]

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        for mod in modules:
            name = mod.replace("/", "_")
            dest = ASSETS / "code" / name
            if dest.exists():
                continue
            try:
                r = await client.get(base + mod)
                r.raise_for_status()
                dest.write_bytes(r.content)
                print(f"  ✓  {name} ({len(r.content)//1024} KB)")
            except Exception as e:
                print(f"  ✗  {name}: {e}")

    print("✓ Code files done\n")


# ──────────────────────────────────────────────
# JSON — synthetic FAQ corpus (Q&A pairs).
# Simulates a help-desk knowledge base:
# the most common RAG use case in production.
# ──────────────────────────────────────────────

def create_faq_json():
    dest = ASSETS / "json" / "faq_corpus.json"
    if dest.exists():
        print("  ↩  faq_corpus.json already exists")
        print("✓ JSON done\n")
        return

    # Realistic FAQ corpus across 4 product domains
    faqs = [
        # Billing
        {"id": "b001", "category": "billing",   "question": "How do I upgrade my subscription plan?",
         "answer": "Go to Settings → Billing → Change Plan. Changes take effect immediately. You are charged the prorated difference.", "tags": ["billing","upgrade","plan"]},
        {"id": "b002", "category": "billing",   "question": "Can I get a refund if I cancel mid-cycle?",
         "answer": "We offer a 7-day money-back guarantee for new subscriptions. After 7 days, cancellations are prorated to the end of the billing cycle.", "tags": ["billing","refund","cancel"]},
        {"id": "b003", "category": "billing",   "question": "What payment methods do you accept?",
         "answer": "We accept Visa, Mastercard, American Express, and PayPal. Bank transfers are available for annual enterprise plans over $5,000.", "tags": ["billing","payment"]},
        {"id": "b004", "category": "billing",   "question": "How do I download my invoice?",
         "answer": "Invoices are available under Settings → Billing → Invoice History. Click any invoice to download the PDF.", "tags": ["billing","invoice"]},
        {"id": "b005", "category": "billing",   "question": "Does the free plan have usage limits?",
         "answer": "Free plan includes 100 API calls/day, 1GB storage, and up to 3 projects. Limits reset at midnight UTC.", "tags": ["billing","free","limits"]},

        # API / Technical
        {"id": "t001", "category": "technical", "question": "What is the rate limit on the API?",
         "answer": "Rate limits depend on plan: Free=10 req/min, Pro=100 req/min, Enterprise=1000 req/min. Exceeding the limit returns a 429 status. Use exponential backoff.", "tags": ["api","rate-limit","technical"]},
        {"id": "t002", "category": "technical", "question": "How do I authenticate API requests?",
         "answer": "Pass your API key in the Authorization header: 'Authorization: Bearer YOUR_API_KEY'. Never include your key in query strings.", "tags": ["api","auth","security"]},
        {"id": "t003", "category": "technical", "question": "Which SDKs do you officially support?",
         "answer": "Official SDKs: Python (pip install yourlib), Node.js (npm install yourlib), Go (go get github.com/yourlib). Community SDKs exist for Ruby and Java.", "tags": ["api","sdk","technical"]},
        {"id": "t004", "category": "technical", "question": "How do I handle pagination in list endpoints?",
         "answer": "All list endpoints return a 'next_cursor' field. Pass it as the 'cursor' query parameter on your next request. An absent cursor means you have reached the last page.", "tags": ["api","pagination","technical"]},
        {"id": "t005", "category": "technical", "question": "What does error code 422 mean?",
         "answer": "HTTP 422 means Unprocessable Entity — your request was valid JSON but failed validation. Check the 'errors' field in the response body for field-level details.", "tags": ["api","errors","technical"]},

        # Account
        {"id": "a001", "category": "account",   "question": "How do I reset my password?",
         "answer": "Click 'Forgot password' on the login page. Enter your email and we will send a reset link valid for 1 hour. Check your spam folder if it doesn't arrive.", "tags": ["account","password"]},
        {"id": "a002", "category": "account",   "question": "Can I change my email address?",
         "answer": "Go to Settings → Profile → Email. Enter your new email and confirm with your password. A verification email is sent to the new address.", "tags": ["account","email","profile"]},
        {"id": "a003", "category": "account",   "question": "How do I enable two-factor authentication?",
         "answer": "Settings → Security → Two-Factor Authentication → Enable. We support TOTP apps (Authy, Google Authenticator) and SMS backup codes.", "tags": ["account","security","2fa"]},
        {"id": "a004", "category": "account",   "question": "How do I delete my account?",
         "answer": "Account deletion is permanent. Go to Settings → Account → Delete Account. All data is removed within 30 days per our data retention policy.", "tags": ["account","delete","privacy"]},
        {"id": "a005", "category": "account",   "question": "Can I transfer my account to another person?",
         "answer": "Account transfers are not self-service. Contact support with both users' emails and we will process the transfer within 2 business days.", "tags": ["account","transfer"]},

        # Product features
        {"id": "f001", "category": "features",  "question": "Does the product support real-time collaboration?",
         "answer": "Yes. Multiple users can edit the same project simultaneously. Changes are synced via WebSockets with a maximum latency of 200ms. Conflict resolution uses operational transforms.", "tags": ["features","collaboration","realtime"]},
        {"id": "f002", "category": "features",  "question": "What export formats are supported?",
         "answer": "Export to: PDF, CSV, JSON, XLSX, Markdown. Exports are processed asynchronously. Large exports (>10k rows) are emailed as a download link.", "tags": ["features","export"]},
        {"id": "f003", "category": "features",  "question": "Is there a mobile app?",
         "answer": "iOS and Android apps are available. The mobile app supports view and comment actions. Creating and editing projects requires the web app.", "tags": ["features","mobile"]},
        {"id": "f004", "category": "features",  "question": "What integrations are available?",
         "answer": "Native integrations: Slack, GitHub, Jira, Notion, Zapier, Make. REST webhooks available on Pro plan and above for custom integrations.", "tags": ["features","integrations"]},
        {"id": "f005", "category": "features",  "question": "Does the product support SSO/SAML?",
         "answer": "SAML 2.0 SSO is available on the Enterprise plan. We support Okta, Azure AD, Google Workspace, and any SAML 2.0 compliant identity provider.", "tags": ["features","sso","enterprise","security"]},

        # Intentional edge cases for analysis
        {"id": "e001", "category": "billing",   "question": "What?", "answer": "Please ask a more specific question.", "tags": []},     # very short — should be flagged
        {"id": "e002", "category": "technical", "question": "How do I upgrade my subscription plan?",  # duplicate of b001
         "answer": "Go to Settings → Billing → Change Plan. Changes take effect immediately. You are charged the prorated difference.", "tags": ["billing","upgrade"]},
        {"id": "e003", "category": "technical", "question": "API authentication",
         "answer": None, "tags": []},   # null answer — should be filtered
    ]

    dest.write_text(json.dumps({"faqs": faqs, "total": len(faqs), "version": "1.0"}, indent=2))
    print(f"  ✓  faq_corpus.json ({len(faqs)} FAQs including edge cases)")
    print("✓ JSON done\n")


# ──────────────────────────────────────────────
# CSV — product catalog with realistic messy data.
# Mix of null values, numeric columns, long text.
# ──────────────────────────────────────────────

def create_product_csv():
    dest = ASSETS / "csv" / "product_catalog.csv"
    if dest.exists():
        print("  ↩  product_catalog.csv already exists")
        print("✓ CSV done\n")
        return

    rows = [
        "id,name,category,description,price_usd,in_stock,tags,rating,review_count",
        '1,Neural Embedder Pro,Software,"Enterprise-grade embedding API supporting 50+ languages. Outputs 1536-dim vectors. Latency <50ms p99. Supports batch and streaming modes.",299.00,true,"nlp,ml,api",4.8,1247',
        '2,VectorStore Cloud,Database,"Managed vector database with automatic sharding. Supports HNSW and IVF-PQ indexes. 99.9% SLA. 100M vectors on entry tier.",149.00,true,"database,vector,cloud",4.6,834',
        '3,RAG Evaluator Kit,Tooling,"Automated evaluation suite for RAG pipelines. Includes faithfulness, relevance, and context recall metrics out of the box.",79.00,true,"eval,rag,testing",4.7,521',
        '4,Document Parser Plus,Tooling,"Handles PDF, Word, HTML, and scanned documents. Layout-aware extraction with table detection. Outputs clean markdown.",49.00,true,"parsing,pdf,ocr",4.5,1089',
        '5,Reranker Turbo,Model,"Cross-encoder reranking model. 2× better precision than BM25. Adds <80ms latency. Supports custom domain fine-tuning.",199.00,true,"reranking,ml,retrieval",4.9,312',
        '6,Chunker Smart,Library,"Semantic chunker with boundary detection. Supports fixed-size, recursive, and topic-based strategies. Python-first.",0.00,true,"chunking,nlp,python",4.4,2103',
        '7,Observability Hub,Platform,"Full-stack RAG observability. Traces every query, retrieval, and generation. Latency breakdown, token cost, hallucination scoring.",129.00,true,"monitoring,observability,devops",4.6,445',
        '8,Hybrid Search Engine,Database,"Combines dense and sparse retrieval with RRF fusion. Outperforms pure dense by 18% on BEIR benchmark.",399.00,false,"search,hybrid,retrieval",4.8,167',
        '9,,,Incomplete row — missing name and category,,,,,',  # messy row
        '10,Context Window Manager,Library,"Dynamically manages context assembly. Handles lost-in-middle, citation tracking, and token budgeting automatically.",0.00,true,"prompting,context,python",4.3,678',
        '11,Multi-modal Embedder,Model,"Embed text and images in a shared vector space. Based on CLIP architecture. Supports batch image ingestion.",349.00,true,"multimodal,vision,embeddings",4.7,234',
        '12,Query Transformer,Library,"HyDE, multi-query, step-back, and decomposition strategies. Plug-and-play with any retriever.",0.00,true,"query,rag,python",4.5,891',
    ]

    dest.write_text("\n".join(rows))
    print(f"  ✓  product_catalog.csv ({len(rows)-1} rows including intentional messy row)")
    print("✓ CSV done\n")


# ──────────────────────────────────────────────
# Pre-fetched web content — avoids hitting live
# servers during analysis. Simulates what WebLoader
# would return after boilerplate stripping.
# ──────────────────────────────────────────────

def create_web_cache():
    articles = {
        "retrieval_augmented_generation.html": {
            "title": "Retrieval-Augmented Generation",
            "url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
            "body": textwrap.dedent("""
                Retrieval-augmented generation (RAG) is a technique for enhancing the accuracy
                and reliability of generative AI models by grounding them on external knowledge
                sources retrieved at inference time.

                RAG combines two components: a retrieval module that searches a knowledge base
                for relevant documents given a query, and a generative module (typically a large
                language model) that synthesizes an answer conditioned on both the query and the
                retrieved context.

                The technique was introduced in a 2020 paper by Lewis et al. at Facebook AI Research.
                It addressed a key limitation of LLMs: factual knowledge is frozen at training time,
                causing models to produce outdated or hallucinated information when asked about
                recent events or specialized domain knowledge.

                Key advantages of RAG over fine-tuning:
                - Knowledge can be updated by updating the retrieval index, without retraining the model.
                - Sources can be cited, enabling attribution and auditability.
                - Domain adaptation is cheaper — curating a retrieval corpus is far less expensive
                  than fine-tuning a large model.

                Variants include Naive RAG (basic retrieve-then-generate), Advanced RAG (query
                transformation, reranking), Modular RAG (configurable pipeline components), and
                Agentic RAG (LLM decides when and what to retrieve).

                Common evaluation metrics include faithfulness (does the answer follow from the
                retrieved context?), answer relevance, context precision, and context recall.
                RAGAS is the most widely adopted evaluation framework.
            """).strip()
        },
        "vector_database.html": {
            "title": "Vector Database",
            "url": "https://en.wikipedia.org/wiki/Vector_database",
            "body": textwrap.dedent("""
                A vector database is a type of database that indexes and stores data as
                high-dimensional vectors, enabling semantic similarity search at scale.

                Unlike traditional databases that search for exact matches, vector databases
                find items that are semantically similar based on distance metrics such as
                cosine similarity, dot product, or Euclidean distance.

                Vector databases became essential infrastructure for AI applications as large
                language models began producing high-quality embeddings — numerical representations
                that capture semantic meaning. When two texts are semantically similar, their
                embedding vectors are close together in the vector space.

                Core components:
                - Embedding storage: Vectors are stored alongside the original content and metadata.
                - Index structures: HNSW (Hierarchical Navigable Small World) and IVF-PQ (Inverted
                  File with Product Quantization) are the dominant index types for approximate
                  nearest neighbor (ANN) search.
                - Filtering: Metadata filters allow hybrid searches (semantic + structured).

                Leading vector databases include Pinecone (managed cloud), Qdrant (open-source,
                Rust-based), Weaviate (multi-modal), Milvus (large-scale), and pgvector
                (PostgreSQL extension for simpler deployments).

                Performance benchmarks typically measure queries per second (QPS), recall at K,
                and index build time. HNSW offers the best recall-latency tradeoff for most
                production RAG workloads.
            """).strip()
        },
        "transformer_architecture.html": {
            "title": "Transformer Architecture",
            "url": "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
            "body": textwrap.dedent("""
                The Transformer is a deep learning architecture introduced in the 2017 paper
                'Attention Is All You Need' by Vaswani et al. It became the foundation of
                nearly all modern large language models.

                The central innovation is the self-attention mechanism, which allows the model
                to weigh the relevance of different positions in an input sequence when computing
                a representation for each position. Unlike recurrent neural networks (RNNs),
                Transformers process all tokens in parallel, making them far more efficient to
                train on modern GPU hardware.

                Architecture components:
                - Token embeddings: Maps each input token to a dense vector.
                - Positional encoding: Injects sequence position information (the model has no
                  inherent sense of order without this).
                - Multi-head self-attention: Runs several attention operations in parallel,
                  each attending to different aspects of the input.
                - Feed-forward network: A two-layer MLP applied independently to each position.
                - Layer normalization: Stabilizes training and enables deeper networks.

                The encoder-decoder structure from the original paper gave rise to BERT-style
                encoders (bidirectional, used for embeddings) and GPT-style decoders (autoregressive,
                used for generation).

                Modern LLMs such as GPT-4, Claude, Gemini, and Llama are all decoder-only
                Transformers scaled to hundreds of billions of parameters.
            """).strip()
        },
        "semantic_search.html": {
            "title": "Semantic Search",
            "url": "https://en.wikipedia.org/wiki/Semantic_search",
            "body": textwrap.dedent("""
                Semantic search is an information retrieval approach that focuses on understanding
                the meaning and intent behind a query rather than relying on keyword matching alone.

                Traditional keyword search systems like BM25 match query terms against document
                terms, missing synonyms, paraphrases, and conceptual relationships. Semantic
                search systems instead encode both the query and documents as dense vectors in a
                shared embedding space, where proximity corresponds to semantic similarity.

                The shift from keyword to semantic search was enabled by large pre-trained language
                models that produce high-quality sentence and document embeddings. Models such as
                Sentence-BERT, E5, and OpenAI's text-embedding-3 series encode text into fixed-size
                vectors that capture deep semantic relationships.

                In production RAG systems, hybrid search — combining dense semantic search with
                sparse keyword search — typically outperforms either approach alone. Reciprocal
                Rank Fusion (RRF) is the standard method for merging ranked lists from both systems.

                Key evaluation datasets for semantic search include BEIR (Benchmarking IR),
                MS MARCO, and Natural Questions. State-of-the-art dense retrievers achieve over
                90% recall@100 on these benchmarks.
            """).strip()
        },
        "prompt_engineering.html": {
            "title": "Prompt Engineering",
            "url": "https://en.wikipedia.org/wiki/Prompt_engineering",
            "body": textwrap.dedent("""
                Prompt engineering is the practice of designing and refining input prompts to
                large language models to elicit desired outputs. It has become a critical skill
                for building reliable AI-powered applications.

                Key prompting techniques:
                - Zero-shot prompting: Ask the model directly without examples. Effective for
                  well-understood tasks where the model has strong prior training signal.
                - Few-shot prompting: Include 2–5 examples of input/output pairs in the prompt.
                  Dramatically improves performance on niche formats and tasks.
                - Chain-of-thought (CoT): Ask the model to reason step by step before giving
                  a final answer. Substantially improves accuracy on multi-step reasoning tasks.
                - System prompts: Instructions placed before the conversation that establish role,
                  behavior, and constraints. Critical for production RAG to define citation format,
                  tone, and out-of-scope behavior.

                In RAG systems, prompt engineering governs how retrieved context is presented to
                the model, how citations are formatted, how the model should handle contradictory
                context, and what to say when the answer is not in the retrieved documents.

                The 'lost in the middle' problem — where LLMs underperform on context placed in
                the middle of a long prompt — is a key concern in RAG prompt design. Best practice
                is to place the most relevant chunks at the beginning and end of the context window.
            """).strip()
        },
    }

    for filename, data in articles.items():
        dest = ASSETS / "web_cache" / filename
        if dest.exists():
            continue
        html = f"""<!DOCTYPE html>
<html><head><title>{data['title']}</title>
<meta name="source-url" content="{data['url']}">
</head><body>
<nav>Wikipedia navigation | Search | Random article</nav>
<h1>{data['title']}</h1>
<div id="content">{data['body']}</div>
<footer>Wikipedia | CC BY-SA | Privacy Policy | Contact</footer>
</body></html>"""
        dest.write_text(html)

    print(f"  ✓  {len(articles)} HTML articles with realistic nav/footer boilerplate")
    print("✓ Web cache done\n")


async def main():
    print("\n📦 RAG Test Data Setup\n" + "─"*40)

    make_dirs()
    print()

    print("📄 PDFs (research papers)...")
    await download_pdfs()

    print("📝 Markdown (project READMEs)...")
    await download_markdown()

    print("🐍 Code (CPython stdlib)...")
    await download_code()

    print("📊 JSON (FAQ corpus)...")
    create_faq_json()

    print("📋 CSV (product catalog)...")
    create_product_csv()

    print("🌐 Web cache (AI topic articles)...")
    create_web_cache()

    # Summary
    total_files = sum(len(list((ASSETS / d).iterdir())) for d in
                      ["pdfs", "markdown", "code", "json", "csv", "web_cache"])
    total_size = sum(f.stat().st_size for f in ASSETS.rglob("*") if f.is_file())

    print("─"*40)
    print(f"✅ Setup complete!")
    print(f"   Total files: {total_files}")
    print(f"   Total size:  {total_size/1024/1024:.1f} MB")
    print()
    print("Next step:")
    print("  python analyze_ingestion.py")


if __name__ == "__main__":
    asyncio.run(main())
