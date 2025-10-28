# DocInsight – Academic Originality Analysis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DocInsight evaluates academic documents against a persistent corpus of real research papers. It combines semantic retrieval (FAISS), optional cross-encoder reranking, stylometry, and document-level aggregation to produce interpretable originality scores and risk spans.

## Features

- **Similarity Fusion**: SBERT retrieval → optional Cross‑Encoder reranking → Stylometry → fused score
- **Adaptive Risk Gating**: Thresholds + semantic floors to suppress weak matches
- **Citation Masking**: Masks citations before scoring
- **Span Clustering & Aggregation**: Consecutive HIGH/MEDIUM grouping + document‑level originality
- **Interpretability**: Match strength labels and gating reasons
- **Persistent Retrieval**: SQLite corpus + FAISS index, reused across runs
- **Multiple Formats**: PDF, DOCX, TXT via robust extractors
- **Streamlit UI**: Upload → metrics dashboard → spans explorer → downloadable reports (HTML/JSON)

## Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │ Enhanced Pipeline│    │ Scoring Engine  │
│                 │───▶│                  │───▶│                 │
│ - Upload/Views  │    │ - Extract/Normalize │ │ - Fusion/Gating │
│ - Reports       │    │ - Retrieval (FAISS) │ │ - Aggregation   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
          │                        │
          ▼                        ▼
   SQLite Corpus           Embeddings + FAISS Index
```

### Key Components
- `config.py`: Configuration and feature flags
- `ingestion/`: arXiv (full‑text PDFs), Wikipedia, web/files; normalization & chunking
- `db/`: SQLite schema and manager
- `embeddings/`: Batch embedding and storage
- `index/`: FAISS / numpy index implementations + manager
- `retrieval/`: Retrieval engine over persistent corpus
- `enhanced_pipeline.py`: End‑to‑end analysis orchestration
- `build_massive_corpus.py`: One‑time corpus builder (~500 papers by default)

## Installation

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt', quiet=True)"
python -m spacy download en_core_web_sm -q
```

## Run

```bash
streamlit run streamlit_app.py
```
Open `http://localhost:8501`, upload a PDF/DOCX/TXT, and view originality analysis.

## Build/Refresh the Corpus (one‑time)

The project can ship with a pre‑built corpus. To (re)build locally (~500 papers):
```bash
action="build" python build_massive_corpus.py
```
This will:
- Ingest arXiv full‑text PDFs across core academic domains + selected Wikipedia articles for breadth
- Generate embeddings for all text chunks
- Build a FAISS index (auto‑reused on subsequent runs)

## Configuration

Key settings in `config.py` include: embedding model, PDF/DOCX extraction limits, chunk sizes and overlap, FAISS index dir, gating thresholds and semantic floors, and feature flags (citation masking, etc.).

## Data & Persistence

- SQLite database `docinsight.db` stores sources, documents, chunks (with embeddings)
- Index files live under `indexes/` (FAISS preferred; numpy fallback when FAISS unavailable)
- Idempotent ingestion: duplicate content (SHA‑256) is skipped automatically
- Embedding & index building are incremental and reused across runs

## Testing

```bash
python -m pytest -q
```

## License

MIT License. See `LICENSE`.