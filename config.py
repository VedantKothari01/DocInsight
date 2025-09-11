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