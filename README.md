# DocInsight - Document Originality Analysis System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DocInsight is an advanced document originality analysis & plagiarism risk detection system that fuses semantic retrieval, cross-encoder reranking, stylometric deviation, and document-level aggregation. The latest iteration adds extended demo corpora, repeated-match decay to prevent a single generic sentence from dominating results, fine-tuned model support, citation masking, and richer interpretability signals.

## 🚀 Features

### Core Capabilities
- **Multi-Layered Similarity Fusion**: Semantic SBERT retrieval → Cross-Encoder reranking → Stylometric proximity → Fused score
- **Adaptive Risk Gating**: Fused thresholds + semantic floor guards (prevents noisy low-semantic matches from inflating risk)
- **Repeated Match Decay** (NEW): Penalizes over-reuse of the same corpus sentence beyond an allowance to reduce generic-sentence inflation
- **Extended Demo Corpus** (NEW): Broader synthetic academic + technical reference base (enable/disable via env)
- **Citation Masking** (NEW): Removes citation artifacts before scoring to avoid false similarity spikes
- **Fine-Tuned Model Support**: Auto-loads locally fine-tuned semantic model if available (`models/semantic_local/`)
- **Span Clustering & Filtering**: Consecutive HIGH/MEDIUM sentences grouped; weak singletons optionally suppressed
- **Document-Level Aggregation**: Coverage + Severity + Span Ratio → Plagiarism Factor → Originality
- **Interpretability Metadata**: Match strength labels (STRONG/MODERATE/WEAK/VERY_WEAK) + gating reasons
- **Interactive Streamlit UI**: Upload, metrics dashboard, spans explorer, downloadable reports (HTML/JSON)
- **Multiple File Formats**: PDF, DOCX, TXT

### Originality Scoring Components
- **Originality Score (0–100%)**: 1 − Plagiarism Factor
- **Plagiarized Coverage**: Token-weighted coverage of HIGH/MEDIUM spans
- **Severity Index**: Token-weighted mean risk score of spans
- **Risk Span Ratio**: Count of (HIGH|MEDIUM) spans ÷ sentence count
- **Plagiarism Factor Breakdown**: Coverage Component + Severity Component + Span Ratio Component with weights α/β/γ
- **Match Strength Label**: Based on normalized semantic score (≥0.75 STRONG, ≥0.55 MODERATE, ≥0.40 WEAK, else VERY_WEAK)

## 🏗️ Architecture

### Phase 1 Implementation (Current)
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │ Enhanced Pipeline│    │ Scoring Engine  │
│                 │───▶│                  │───▶│                 │
│ - File Upload   │    │ - Text Extract   │    │ - Sentence Cls  │
│ - Metrics View  │    │ - Preprocessing  │    │ - Span Cluster  │
│ - Risk Spans    │    │ - Multi-analysis │    │ - Doc Scoring   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌────────▼────────┐
                       │ ML Components   │
                       │                 │
                       │ - SBERT Search  │
                       │ - Cross Encoder │
                       │ - Stylometry    │
                       │ - FAISS Index   │
                       └─────────────────┘
```

### Key Components
- **`config.py`**: Thresholds, weights, feature flags (extended corpus, fine-tune, citation masking)
- **`scoring/core.py`**: Fused sentence scoring, semantic floors & gating, span clustering, document aggregation
- **`enhanced_pipeline.py`**: End-to-end orchestrator: text extract → sentence embedding → cross-encode rerank → stylometry → fusion → risk classification → reuse decay → aggregation
- **`streamlit_app.py`**: UI with status indicators (extended corpus flag, breakdowns, spans) & report generation
- **`embeddings/`, `index/`**: Embedding model loading and FAISS / fallback index management
- **`ingestion/`**: Early loaders (arxiv/wiki/web/file) for future phases
- **`fine_tuning/`**: Scripts for dataset prep & model fine-tuning/evaluation
- **`scripts/`**: Evaluation & fine-tune orchestration utilities

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### 🚀 One-Command Setup & Run
```bash
# Clone the repository
git clone https://github.com/VedantKothari01/DocInsight.git
cd DocInsight

# Install dependencies and run DocInsight in one command
bash run_docinsight.sh
```

The script automatically:
- ✅ Verifies Python 3.8+ requirement
- 📦 Installs all dependencies from requirements.txt
- 📚 Downloads required NLTK data
- 🧠 Sets up spaCy language model
- 🔍 Tests core imports
- 🌐 Starts the Streamlit web interface

### Manual Setup (Alternative)
```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data (first run only)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the application
streamlit run streamlit_app.py
```

### Dependencies (Major)
- **sentence-transformers** (SBERT + CrossEncoder)
- **transformers** (model backend)
- **faiss-cpu** (if available; otherwise numpy fallback auto-engaged)
- **spaCy** + NLTK (linguistic + sentence segmentation)
- **PyMuPDF / docx2txt** (extraction)
- **streamlit** (UI)
- **numpy, scikit-learn, torch** (core ML math)

## 🎯 Usage

### 🚀 Quick Start (Recommended)
```bash
# One command to run everything
bash run_docinsight.sh
```
This will:
1. Install all dependencies
2. Set up required data files
3. Start the web interface at `http://localhost:8501`
4. Ready to analyze documents!

### Web Interface
1. Start the Streamlit application:
   ```bash
   streamlit run streamlit_app.py
   ```
2. Open your browser to `http://localhost:8501`
3. Upload a document (PDF, DOCX, or TXT)
4. View originality analysis and download reports

### Programmatic Usage
```python
from enhanced_pipeline import DocumentAnalysisPipeline

# Initialize pipeline
pipeline = DocumentAnalysisPipeline()

# Analyze document
result = pipeline.analyze_document('document.pdf')

# Access originality metrics
metrics = result['originality_analysis']['originality_metrics']
print(f"Originality Score: {metrics['originality_score']:.1%}")
print(f"Risk Spans: {result['originality_analysis']['total_risk_spans']}")
```

## 📊 Originality Scoring Methodology

### Aggregation Formula
```
Originality Score = 1 - ( α·Coverage + β·Severity + γ·Span_Ratio )
Default Weights: α=0.55, β=0.30, γ=0.15
```

### Scoring Components
1. **Coverage**: Token-weighted percentage of document covered by risk spans
2. **Severity**: Average similarity scores of identified risk spans
3. **Span Ratio**: Proportion of sentences forming risk spans

### Risk Classification & Semantic Floors
Risk level requires BOTH a fused score threshold and semantic normalized minimum (floors reduce noisy matches):

| Level | Fused Threshold | Semantic Norm Floor | Notes |
|-------|-----------------|---------------------|-------|
| HIGH  | ≥ 0.70          | ≥ 0.60              | Requires strong fused + semantic alignment |
| MED   | ≥ 0.40          | ≥ 0.40              | Moderate similarity; below HIGH floor |
| LOW   | else            | n/a                 | Or semantic raw < 0.35 always forces LOW |

Semantic raw score < 0.35 forcibly downgrades to LOW (minimum evidence guard).

## 🔧 Configuration

Key settings in `config.py`:

### Core Flags & Environment Variables
| Purpose | Variable | Default | Effect |
|---------|----------|---------|--------|
| Use fine-tuned semantic model | `DOCINSIGHT_USE_FINE_TUNED` | `true` | Load `models/semantic_local/` if present |
| Force re-train fine-tuned model | `DOCINSIGHT_FORCE_RETRAIN` | `false` | Rebuild fine-tuned model artifacts |
| Enable extended demo corpus | `DOCINSIGHT_EXTENDED_CORPUS` | `true` | Adds broader multi-domain reference set |
| Enable citation masking | `DOCINSIGHT_CITATION_MASKING_ENABLED` | `true` | Masks citations before scoring |
| Adjust semantic weight | `DOCINSIGHT_W_SEMANTIC` | `0.6` | Fusion weight (semantic) |
| Adjust stylometry weight | `DOCINSIGHT_W_STYLO` | `0.25` | Fusion weight (stylometry) |
| Adjust AI-likeness weight | `DOCINSIGHT_W_AI` | `0.15` | Fusion weight (AI-likeness) |
| Min semantic raw match | `SEMANTIC_MIN_MATCH` | `0.35` (code) | Below → forced LOW risk |

Additional risk gating floors defined in code: `SEMANTIC_HIGH_FLOOR=0.60`, `SEMANTIC_MEDIUM_FLOOR=0.40`.

### Repeated Match Decay (Post-Processing)
Applied after per-sentence scoring: repeated reuse of the same corpus sentence beyond an allowance (currently 2) is multiplicatively decayed (factor 0.85 per extra occurrence) and can downgrade risk. Future: expose parameters via config.

### Outputs & Interpretability
- Each sentence now includes: `risk_level`, `confidence_score` (fused), `match_strength`, `reason`, top match text.
- Document metrics include plagiarism factor component breakdown.

## 🛣️ Development Roadmap

### Phase 2: Database & Ingestion Pipeline (Planned)
- [ ] SQLite/FAISS hybrid storage
- [ ] Modular document loaders
- [ ] Large-scale corpus management
- [ ] Advanced indexing strategies (IVF/PQ)
- [ ] Real-time corpus updates

### Phase 3: Model Fine-tuning & Evaluation (Planned)
- [ ] Custom SBERT fine-tuning on academic datasets
- [ ] Cross-encoder training for domain-specific tasks
- [ ] Stylometry classifier improvements
- [ ] Evaluation metrics and benchmarking
- [ ] Learned fusion model (replacing heuristic weights)

### Phase 4: Advanced Features (Future)
- [ ] Multi-language support
- [ ] Citation detection and analysis
- [ ] Academic writing assessment
- [ ] Integration APIs
- [ ] Advanced visualization

## 🧪 Testing

### Validation Checklist (Phase 1 End)
- [x] Dependency installation succeeds on clean env
- [x] Streamlit UI loads and processes sample docs (TXT/PDF/DOCX)
- [x] Extended corpus loads (flag visible in sidebar)
- [x] Repeated match decay reduces duplicate HIGH flags
- [x] Citation masking applied (logs show mask counts when enabled)
- [x] Originality score + component breakdown visible
- [x] Risk spans clustered & weak singletons filtered
- [x] Reports (HTML/JSON) generated

### Focused Signature Update
`SentenceClassifier.classify_sentence` now returns 4-tuple:
```
(risk_level, fused_score, match_strength, reason)
```
Tests updated accordingly.

### Quick Programmatic Smoke Test
```bash
python - <<'PY'
from enhanced_pipeline import DocumentAnalysisPipeline
p = DocumentAnalysisPipeline()
r = p.analyze_document('sample_document.txt')
m = r['originality_analysis']['originality_metrics']
print('Originality:', f"{m['originality_score']:.1%}")
print('Coverage:', f"{m['plagiarized_coverage']:.1%}")
print('Spans:', r['originality_analysis']['total_risk_spans'])
PY
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Sentence-BERT for semantic embeddings
- Hugging Face Transformers for cross-encoder models
- FAISS for efficient similarity search
- spaCy for natural language processing
- Streamlit for the web interface

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the demo notebook for examples

---

**DocInsight** – Advancing document originality analysis through multi-signal fusion & interpretability