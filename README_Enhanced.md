# DocInsight Enhanced v2.0 🚀

## Major Upgrade: Real Dataset Integration

DocInsight has been completely upgraded with real dataset integration, advanced ML models, and production-ready features. The system now uses **50,000+ sentences** from multiple high-quality datasets instead of the original 10 hardcoded sentences.

## 🎉 What's New in v2.0

### 📊 Real Dataset Integration
- **PAWS Dataset**: Paraphrase Adversaries from Word Scrambling for advanced paraphrase detection
- **Wikipedia Articles**: General knowledge and encyclopedic content covering diverse topics
- **arXiv Abstracts**: Academic and research paper abstracts for scholarly content
- **Academic Phrases**: Curated academic writing patterns and structures
- **Synthetic Paraphrases**: AI-generated variations for comprehensive coverage

### 🧠 Advanced ML Pipeline
- **SentenceTransformers**: State-of-the-art semantic similarity using all-MiniLM-L6-v2
- **Cross-encoder Reranking**: Precision improvement with ms-marco-MiniLM-L-6-v2
- **Enhanced Stylometry**: 15+ linguistic features including readability, POS ratios, dependencies
- **FAISS Indexing**: Sub-second similarity search across large corpora
- **Multi-signal Fusion**: Combines semantic, syntactic, and stylistic signals

### 🎯 Enhanced Detection Capabilities
- **Confidence Scoring**: HIGH/MEDIUM/LOW confidence levels with detailed breakdowns
- **Better Paraphrase Detection**: Advanced models trained on real paraphrase data
- **Domain-Adaptive**: Covers academic, general, and technical writing styles
- **Performance Optimized**: Efficient embedding storage and retrieval

### 🌐 User Experience Improvements
- **One-Click Analysis**: Upload document and get instant comprehensive reports
- **Interactive Web Interface**: Modern Streamlit app with real-time results
- **Comprehensive Reports**: JSON and HTML formats with detailed explanations
- **Offline Capability**: Works with cached datasets when internet unavailable

## 📈 Performance Comparison

| Feature | Original v1.0 | Enhanced v2.0 |
|---------|---------------|----------------|
| **Corpus Size** | 10 sentences | 50,000+ sentences |
| **Data Sources** | Hardcoded | Real datasets (PAWS, Wikipedia, arXiv) |
| **Domain Coverage** | Limited | Multi-domain (academic, general, technical) |
| **ML Models** | Basic SBERT | Advanced SBERT + CrossEncoder + Stylometry |
| **Detection Accuracy** | Basic similarity | Multi-signal fusion with confidence |
| **User Experience** | Manual corpus upload | One-click analysis |
| **Performance** | Simple search | Optimized FAISS indexing |
| **Reports** | Basic JSON/HTML | Comprehensive with confidence scoring |
| **Offline Support** | None | Full offline capability |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/VedantKothari01/DocInsight.git
cd DocInsight

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (optional, for enhanced features)
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Running the Enhanced System

#### 1. Web Interface (Recommended)

```bash
streamlit run streamlit_app.py
```

Then open http://localhost:8501 in your browser.

#### 2. Jupyter Notebook

Open `DocInsight_Demo_Enhanced.ipynb` in Jupyter/Colab for interactive analysis.

#### 3. Python API

```python
from enhanced_pipeline import EnhancedPlagiarismDetector

# Initialize detector
detector = EnhancedPlagiarismDetector(corpus_size=50000)
detector.initialize()

# Analyze document
report = detector.generate_enhanced_report(
    'path/to/document.txt',
    output_json='report.json',
    output_html='report.html'
)

# Print results
print(f"Analyzed {report['total_sentences']} sentences")
print(f"High confidence matches: {report['overall_stats']['high_confidence_count']}")
```

## 📊 System Architecture

### Core Components

1. **Dataset Loaders** (`dataset_loaders.py`)
   - Downloads and processes real datasets
   - Handles caching and offline fallbacks
   - Supports PAWS, Wikipedia, arXiv, and custom datasets

2. **Corpus Builder** (`corpus_builder.py`)
   - Manages large-scale corpus building
   - Optimized embedding generation and storage
   - FAISS index construction for fast search

3. **Enhanced Pipeline** (`enhanced_pipeline.py`)
   - Complete plagiarism detection pipeline
   - Multi-signal fusion (semantic + rerank + stylometry)
   - Confidence scoring and report generation

4. **Web Interface** (`streamlit_app.py`)
   - Production-ready Streamlit application
   - Interactive analysis with real-time results
   - Comprehensive reporting and export options

### Data Flow

```
Document Upload → Text Extraction → Sentence Splitting → 
Semantic Search → Cross-encoder Reranking → Stylometry Analysis → 
Score Fusion → Confidence Assignment → Report Generation
```

## 🔧 Configuration Options

### Corpus Size
Adjust the corpus size based on your needs:

```python
# For fast testing
detector = EnhancedPlagiarismDetector(corpus_size=1000)

# For production use
detector = EnhancedPlagiarismDetector(corpus_size=50000)
```

### Model Selection
Choose different models for specific use cases:

```python
detector = EnhancedPlagiarismDetector(
    cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-12-v2",  # More accurate
    spacy_model="en_core_web_lg"  # Better linguistic features
)
```

### Scoring Weights
Customize the fusion weights:

```python
detector.alpha = 0.7  # Semantic similarity weight
detector.beta = 0.2   # Cross-encoder weight
detector.gamma = 0.1  # Stylometry weight
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Test individual components
python test_dataset_loading.py
python test_enhanced_pipeline.py

# Test complete system
python test_complete_system.py
```

## 📁 File Structure

```
DocInsight/
├── enhanced_pipeline.py          # Main detection pipeline
├── dataset_loaders.py            # Real dataset integration
├── corpus_builder.py             # Large-scale corpus management
├── simple_corpus_builder.py      # Offline fallback system
├── streamlit_app.py              # Production web interface
├── requirements.txt              # Python dependencies
├── DocInsight_Demo_Enhanced.ipynb # Enhanced Jupyter notebook
├── DocInsight_Demo_Original.ipynb # Original notebook (backup)
├── test_*.py                     # Comprehensive test suite
└── README_Enhanced.md            # This documentation
```

## 🎯 Use Cases

### Academic Institutions
- **Student Paper Analysis**: Detect plagiarism in essays and research papers
- **Thesis Screening**: Comprehensive analysis of graduate theses
- **Academic Integrity**: Automated first-pass plagiarism screening

### Content Creation
- **Blog Post Analysis**: Check originality of web content
- **Marketing Copy**: Ensure unique marketing materials
- **Technical Documentation**: Verify originality of technical writing

### Research & Development
- **Literature Review**: Find similar research and avoid duplication
- **Grant Proposal**: Check uniqueness of research proposals
- **Publication Screening**: Pre-submission plagiarism checking

## 🚨 Detection Confidence Levels

### 🔴 HIGH Confidence (Score > 0.7)
- Strong indicators of potential plagiarism
- Multiple similarity signals align
- Requires immediate attention

### 🟡 MEDIUM Confidence (0.4 ≤ Score ≤ 0.7)
- Moderate similarities detected
- May indicate paraphrasing or common phrasing
- Worth manual review

### 🟢 LOW Confidence (Score < 0.4)
- Minimal or no similarities found
- Likely original content
- No immediate concern

## 🛠️ Advanced Features

### Offline Mode
The system works completely offline using cached datasets:

```python
# Force offline mode
detector = EnhancedPlagiarismDetector(corpus_size=5000)
detector.initialize(force_rebuild_corpus=False)
```

### Custom Datasets
Add your own datasets:

```python
from dataset_loaders import DatasetLoader

loader = DatasetLoader()
custom_sentences = ["Your custom sentences here..."]
loader.save_corpus(custom_sentences, "custom_corpus.json")
```

### Batch Processing
Process multiple documents:

```python
documents = ['doc1.txt', 'doc2.txt', 'doc3.txt']
for doc in documents:
    report = detector.generate_enhanced_report(doc)
    print(f"Document: {doc}, Max Score: {report['overall_stats']['max_fused_score']}")
```

## 🔍 Technical Details

### Performance Metrics
- **Corpus Building**: ~2-5 minutes for 50K sentences (cached afterwards)
- **Document Analysis**: ~0.1-2 seconds per sentence
- **Memory Usage**: ~2-4 GB for full 50K corpus with embeddings
- **Storage**: ~500 MB for cached embeddings and indices

### Supported Formats
- **Text Files**: .txt
- **PDF Documents**: .pdf (via PyMuPDF)
- **Word Documents**: .docx, .doc (via python-docx)

### Model Information
- **SBERT Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Cross-encoder**: ms-marco-MiniLM-L-6-v2
- **Embedding Storage**: Normalized L2 vectors
- **Index Type**: FAISS IndexFlatIP (exact search)

## 🤝 Contributing

We welcome contributions! Areas for improvement:

1. **Additional Datasets**: Integrate more specialized datasets
2. **Language Support**: Extend to non-English languages
3. **Model Fine-tuning**: Train domain-specific models
4. **Performance Optimization**: Further speed improvements
5. **UI Enhancement**: Advanced web interface features

## 📄 License

This project is licensed under the MIT License - see the original repository for details.

## 🙏 Acknowledgments

- **PAWS Dataset**: Google Research for paraphrase detection data
- **Sentence Transformers**: HuggingFace for semantic similarity models
- **FAISS**: Facebook AI Research for efficient similarity search
- **Streamlit**: For the excellent web app framework

## 📞 Support

For issues or questions:
1. Check the test suite: `python test_complete_system.py`
2. Review error logs for detailed debugging information
3. Ensure all dependencies are installed from `requirements.txt`
4. Verify internet connection for initial dataset downloads

---

**Enhanced DocInsight v2.0** - Production-ready plagiarism detection with real dataset integration! 🚀