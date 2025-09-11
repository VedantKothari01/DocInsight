# DocInsight - Document Originality Analysis System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DocInsight is an advanced document analysis system that detects potential plagiarism and evaluates document originality using state-of-the-art natural language processing techniques.

## 🚀 Features

### Core Capabilities
- **Multi-layered Similarity Detection**: Combines semantic search, cross-encoder reranking, and stylometric analysis
- **Document-level Aggregation**: Provides comprehensive originality scores and metrics
- **Risk Span Clustering**: Groups consecutive similar sentences into coherent spans
- **Interactive Web Interface**: User-friendly Streamlit app for document upload and analysis
- **Multiple File Formats**: Supports PDF, DOCX, and TXT files

### Originality Scoring
- **Originality Score (0-100%)**: Overall document originality assessment
- **Plagiarized Coverage**: Percentage of document content identified as potentially plagiarized
- **Severity Index**: Weighted average of similarity scores across risk spans
- **Risk Span Analysis**: Detailed breakdown of suspicious content areas

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
- **config.py**: Centralized configuration management
- **scoring.py**: Sentence classification and document-level scoring algorithms
- **enhanced_pipeline.py**: Main analysis pipeline with integrated scoring
- **streamlit_app.py**: Modern web interface with originality metrics

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/VedantKothari01/DocInsight.git
cd DocInsight

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (first run only)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Run the application
streamlit run streamlit_app.py
```

### Dependencies
- **sentence-transformers**: Semantic similarity search
- **transformers**: Cross-encoder reranking
- **faiss-cpu**: Efficient similarity search indexing
- **spacy**: Natural language processing and stylometry
- **streamlit**: Web interface framework
- **PyMuPDF**: PDF text extraction
- **docx2txt**: DOCX text extraction

## 🎯 Usage

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
Originality = 1 - f(coverage, severity, span_ratio)

where:
f(coverage, severity, span_ratio) = α×coverage + β×severity + γ×span_ratio

Default weights:
- α (coverage) = 0.55
- β (severity) = 0.30  
- γ (span_ratio) = 0.15
```

### Scoring Components
1. **Coverage**: Token-weighted percentage of document covered by risk spans
2. **Severity**: Average similarity scores of identified risk spans
3. **Span Ratio**: Proportion of sentences forming risk spans

### Risk Classification
- **HIGH (🔴)**: Fused similarity score ≥ 0.7
- **MEDIUM (🟡)**: Fused similarity score ≥ 0.4
- **LOW (🟢)**: Fused similarity score < 0.4

## 🔧 Configuration

Key settings in `config.py`:

```python
# Model configurations
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'
CROSS_ENCODER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

# Scoring thresholds
HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.4

# Aggregation weights
AGGREGATION_WEIGHTS = {
    'alpha': 0.55,  # Coverage weight
    'beta': 0.30,   # Severity weight
    'gamma': 0.15   # Span ratio weight
}
```

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

### Validation Checklist
- [x] pip install succeeds from requirements.txt in clean environment
- [x] streamlit run loads application successfully
- [x] Processes sample documents (.txt/.pdf/.docx)
- [x] Originality score displays with non-zero values on mixed content
- [x] High/Medium risk spans display when similarity patterns exist
- [x] No unwanted caches or logs committed to repository

### Sample Test
```bash
# Run with demo document
python -c "
from enhanced_pipeline import analyze_document_file
result = analyze_document_file('sample_document.txt')
print(f'Originality: {result[\"originality_analysis\"][\"originality_metrics\"][\"originality_score\"]:.1%}')
"
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

**DocInsight** - Advancing document originality analysis through state-of-the-art NLP