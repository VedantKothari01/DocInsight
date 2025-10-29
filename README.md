# DocInsight – Academic Originality Analysis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DocInsight evaluates academic documents against a persistent corpus of real research papers. It combines semantic retrieval (FAISS), optional cross-encoder reranking, stylometry, and document-level aggregation to produce interpretable originality scores and risk spans.

## Features

- **Batch Processing**: Upload and analyze up to 15 documents simultaneously
- **Similarity Fusion**: SBERT retrieval → optional Cross‑Encoder reranking → Stylometry → fused score
- **Adaptive Risk Gating**: Thresholds + semantic floors to suppress weak matches
- **Citation Masking**: Masks citations before scoring
- **Span Clustering & Aggregation**: Consecutive HIGH/MEDIUM grouping + document‑level originality
- **Interpretability**: Match strength labels and gating reasons
- **Persistent Retrieval**: SQLite corpus + FAISS index, reused across runs
- **Multiple Formats**: PDF, DOCX, TXT via robust extractors
- **Streamlit UI**: Multi-file upload → batch summary → individual metrics dashboard → spans explorer → downloadable reports (HTML/JSON)

## Architecture
```
┌──────────────────────┐
│   Streamlit UI       │
│  - Multi-Upload      │
│  - Batch Summary     │
│  - Progress Tracking │
│  - Reports           │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Enhanced Pipeline    │
│  - Extract/Normalize │
│  - Citation Masking  │
│  - Retrieval (FAISS) │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Scoring Engine     │
│  - Semantic Fusion   │
│  - Cross-Encoder     │
│  - Stylometry        │
│  - Risk Gating       │
│  - Aggregation       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Data Layer          │
│  - SQLite Corpus     │
│  - FAISS Index       │
│  - Fine-tuned Models │
└──────────────────────┘
```

### Key Components
- `config.py`: Configuration and feature flags
- `ingestion/`: arXiv (full‑text PDFs), Wikipedia, web/files; normalization & chunking
- `db/`: SQLite schema and manager
- `embeddings/`: Batch embedding and storage
- `index/`: FAISS / numpy index implementations + manager
- `retrieval/`: Retrieval engine over persistent corpus
- `enhanced_pipeline.py`: End‑to‑end analysis orchestration
- `streamlit_app.py`: Multi-file upload UI with batch processing
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
Open `http://localhost:8501`, upload one or multiple PDF/DOCX/TXT files (up to 15 at once), and view originality analysis.

### Workflow

1. **Upload Documents**: Select up to 15 files simultaneously
2. **Batch Analysis**: Click "Analyze All Documents" to process all files
3. **Summary View**: Review comparative metrics across all documents
4. **Individual Reports**: Select any document for detailed sentence-level analysis
5. **Download**: Export HTML/JSON reports for each document

## Build/Refresh the Corpus (one‑time)

The project can ship with a pre‑built corpus. To (re)build locally (~500 papers):
```bash
action="build" python build_massive_corpus.py
```
This will:
- Ingest arXiv full‑text PDFs across core academic domains + selected Wikipedia articles for breadth
- Generate embeddings for all text chunks
- Build a FAISS index (auto‑reused on subsequent runs)

Alternatively, use the built-in "Build Academic Corpus" button in the Streamlit sidebar for a starter corpus (~20 documents across multiple domains).

## Configuration

Key settings in `config.py` include:
- Embedding model selection
- PDF/DOCX extraction limits
- Chunk sizes and overlap
- FAISS index directory
- Gating thresholds and semantic floors
- Feature flags (citation masking, batch processing limits, etc.)
- `MAX_FILES = 15` controls batch upload limit

## Data & Persistence

- SQLite database `docinsight.db` stores sources, documents, chunks (with embeddings)
- Index files live under `indexes/` (FAISS preferred; numpy fallback when FAISS unavailable)
- `models/` folder contains fine-tuned models optimized for academic plagiarism detection
- Idempotent ingestion: duplicate content (SHA‑256) is skipped automatically
- Embedding & index building are incremental and reused across runs
- Analysis results cached per document for fast re-analysis

## Batch Processing Features

- **Progress Tracking**: Real-time progress bar and status updates during batch analysis
- **Error Handling**: Individual file failures don't stop the entire batch
- **Comparative Summary**: Side-by-side metrics for all analyzed documents
- **Cached Results**: Re-analyzing the same file uses cached results
- **Individual Reports**: Separate downloadable reports for each document

## Testing

```bash
python -m pytest -q
```

## License

MIT License. See `LICENSE`.