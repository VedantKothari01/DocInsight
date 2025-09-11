"""
Configuration module for DocInsight

Centralized constants for model names, thresholds, paths, and aggregation weights.
"""

# Model configurations
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'
CROSS_ENCODER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
SPACY_MODEL_NAME = 'en_core_web_sm'

# Scoring thresholds
HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.4
MIN_SENTENCE_LENGTH = 3

# Additional semantic similarity floors for more robust risk gating
# A candidate must satisfy BOTH fused thresholds (above) and these semantic floors
# to be promoted to a higher risk class. Prevents random noisy matches.
SEMANTIC_HIGH_FLOOR = 0.60   # Minimum normalized semantic score to allow HIGH
SEMANTIC_MEDIUM_FLOOR = 0.40 # Minimum normalized semantic score to allow MEDIUM
SEMANTIC_MIN_MATCH = 0.35    # Below this semantic raw score => treat as LOW regardless

# Aggregation weights for document-level scoring
# Formula: Originality = 1 - f(coverage, severity, span_ratio)
AGGREGATION_WEIGHTS = {
    'alpha': 0.55,  # Coverage weight - how much of document is covered by plagiarized spans
    'beta': 0.30,   # Severity weight - average severity of plagiarized spans  
    'gamma': 0.15   # Span ratio weight - ratio of plagiarized spans to total spans
}

# Fusion weights for sentence-level scoring (semantic, cross-encoder, stylometry)
FUSION_WEIGHTS = {
    'semantic': 0.6,
    'cross_encoder': 0.3, 
    'stylometry': 0.1
}

# Search parameters
DEFAULT_TOP_K = 5
MAX_CANDIDATES = 10

# File paths and extensions
SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.docx']
TEMP_DIR = '/tmp'

# UI configuration
MAX_SENTENCE_DISPLAY = 100  # Limit sentence detail output to prevent UI overload
TOP_RISK_SPANS_PREVIEW = 3  # Number of top risk spans to show in preview

# Cache and index configuration
INDEX_DIMENSION = 384  # SBERT all-MiniLM-L6-v2 embedding dimension
FAISS_INDEX_TYPE = 'IndexFlatIP'  # Inner product for cosine similarity

# Stylometry normalization constants
FLESCH_SCORE_NORMALIZATION = 50.0

# Risk level mappings
RISK_LEVELS = {
    'HIGH': 'HIGH',
    'MEDIUM': 'MEDIUM', 
    'LOW': 'LOW'
}

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Phase 2: Database and Indexing Configuration
DB_PATH = 'docinsight.db'
INDEX_PATH = 'indexes/'
EMBEDDING_MODEL = SBERT_MODEL_NAME  # Reuse existing SBERT model
MAX_CHUNKS_PER_DOC = 100
CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP = 50
RETRIEVAL_TOP_K = 10
MIN_SIM_THRESHOLD = 0.3
INDEX_TYPE = 'IndexFlatIP'  # FAISS index type
INGEST_BATCH_SIZE = 32

# Ingestion source limits
ARXIV_RATE_LIMIT = 1.0  # seconds between requests
WIKI_MAX_PAGES = 50
MAX_WEB_CONCURRENCY = 5

# File processing
PDF_MAX_PAGES = 100
DOCX_MAX_PARAGRAPHS = 500

# Chunking strategy
CHUNKING_STRATEGY = 'sentence'  # 'sentence' or 'sliding_window'
MIN_CHUNK_LENGTH = 50  # minimum characters per chunk
MAX_CHUNK_LENGTH = 2000  # maximum characters per chunk

# Language filtering
SUPPORTED_LANGUAGES = ['en']  # Language codes to keep during ingestion
LANGUAGE_DETECTION_CONFIDENCE = 0.8

# Phase 2+: Additional Configuration for Fine-tuning, Stylometry, and Enhanced Scoring
import os

# Model paths and configuration
MODEL_BASE_NAME = SBERT_MODEL_NAME  # Base model for semantic similarity
MODEL_FINE_TUNED_PATH = 'models/semantic_local/'  # Path for fine-tuned model
USE_FINE_TUNED_MODEL = os.getenv('DOCINSIGHT_USE_FINE_TUNED', 'true').lower() == 'true'
FORCE_RETRAIN = os.getenv('DOCINSIGHT_FORCE_RETRAIN', 'false').lower() == 'true'
EXTENDED_CORPUS_ENABLED = os.getenv('DOCINSIGHT_EXTENDED_CORPUS', 'true').lower() == 'true'
AI_LIKENESS_MODEL_PATH = 'models/ai_likeness/'  # Path for AI-likeness classifier

# Document processing configuration
CHUNK_SIZE = int(os.getenv('DOCINSIGHT_CHUNK_SIZE', '512'))  # Tokens per chunk
OVERLAP = int(os.getenv('DOCINSIGHT_OVERLAP', '50'))  # Overlap between chunks
SECTION_MIN_TOKENS = int(os.getenv('DOCINSIGHT_SECTION_MIN_TOKENS', '100'))  # Minimum tokens per section

# Citation masking configuration
CITATION_MASKING_ENABLED = os.getenv('DOCINSIGHT_CITATION_MASKING_ENABLED', 'true').lower() == 'true'

# Scoring weights (with environment variable overrides)
# Rationale: Semantic similarity provides the strongest signal for plagiarism detection,
# stylometry helps identify writing pattern deviations, and AI-likeness detects
# potential AI-generated content that may indicate sophisticated plagiarism attempts.
WEIGHT_SEMANTIC = float(os.getenv('DOCINSIGHT_W_SEMANTIC', '0.6'))  # Semantic similarity weight
WEIGHT_STYLO = float(os.getenv('DOCINSIGHT_W_STYLO', '0.25'))  # Stylometric deviation weight  
WEIGHT_AI = float(os.getenv('DOCINSIGHT_W_AI', '0.15'))  # AI-likeness probability weight

# Ensure weights sum to 1.0 (normalize if environment overrides don't sum to 1.0)
_total_weight = WEIGHT_SEMANTIC + WEIGHT_STYLO + WEIGHT_AI
if abs(_total_weight - 1.0) > 0.01:  # Allow small floating point errors
    WEIGHT_SEMANTIC = WEIGHT_SEMANTIC / _total_weight
    WEIGHT_STYLO = WEIGHT_STYLO / _total_weight
    WEIGHT_AI = WEIGHT_AI / _total_weight

# Academic document section patterns
ACADEMIC_SECTIONS = {
    'abstract': ['abstract', 'summary'],
    'introduction': ['introduction', 'intro'],
    'methods': ['methods', 'methodology', 'approach', 'materials and methods'],
    'results': ['results', 'findings'],
    'discussion': ['discussion', 'analysis'],
    'conclusion': ['conclusion', 'conclusions', 'summary'],
    'references': ['references', 'bibliography', 'works cited']
}

# Citation patterns for masking
CITATION_PATTERNS = {
    'numeric': [r'\[\d+\]', r'\(\d+\)', r'\d+\s*\)'],  # [1], (1), 1)
    'author_year': [r'\([A-Za-z]+\s*et\s*al\.?\s*,?\s*\d{4}\)', r'\([A-Za-z]+\s*,?\s*\d{4}\)'],  # (Smith et al., 2020)
    'footnote': [r'\d+\s*\)', r'^\d+\s'],  # Footnote numbers
}

# Fine-tuning configuration
FINE_TUNING_EPOCHS = int(os.getenv('DOCINSIGHT_FINE_TUNING_EPOCHS', '3'))
FINE_TUNING_BATCH_SIZE = int(os.getenv('DOCINSIGHT_FINE_TUNING_BATCH_SIZE', '16'))
FINE_TUNING_LEARNING_RATE = float(os.getenv('DOCINSIGHT_FINE_TUNING_LR', '2e-5'))

# Stylometry feature configuration
FUNCTION_WORDS = [
    'the', 'of', 'and', 'a', 'to', 'in', 'is', 'you', 'that', 'it', 
    'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'i'
]  # Most common function words for stylometric analysis

# AI-likeness detection thresholds
AI_LIKENESS_THRESHOLD = float(os.getenv('DOCINSIGHT_AI_THRESHOLD', '0.7'))
SUSPICIOUS_SECTION_COUNT = int(os.getenv('DOCINSIGHT_SUSPICIOUS_SECTIONS', '3'))  # Top N suspicious sections to show