-- DocInsight Phase 2 Database Schema
-- SQLite database for persistent corpus storage

-- Sources table: Track different ingestion sources
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,  -- 'file', 'web', 'wiki', 'arxiv', etc.
    locator TEXT NOT NULL,  -- file path, URL, etc.
    meta_json TEXT,  -- JSON metadata specific to source type
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Documents table: Individual documents from sources
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    title TEXT,
    raw_path TEXT,  -- original file path or URL
    url TEXT,  -- canonical URL if applicable
    content_hash TEXT NOT NULL,  -- SHA256 of content for deduplication
    char_count INTEGER DEFAULT 0,
    language TEXT DEFAULT 'en',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
);

-- Chunks table: Text chunks with embeddings
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,  -- order within document
    text TEXT NOT NULL,
    token_count INTEGER DEFAULT 0,
    char_count INTEGER DEFAULT 0,
    embedding BLOB,  -- serialized numpy array (nullable until embedded)
    hash TEXT NOT NULL,  -- SHA256 of text for deduplication
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- Ingestion runs table: Track batch ingestion operations
CREATE TABLE IF NOT EXISTS ingestion_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at TIMESTAMP,
    source_count INTEGER DEFAULT 0,
    doc_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'running',  -- 'running', 'completed', 'failed'
    log_path TEXT,
    error_message TEXT
);

-- Settings table: Key-value configuration storage
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_id);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(hash);
CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(type);

-- Insert default settings
INSERT OR IGNORE INTO settings (key, value, description) VALUES
    ('schema_version', '1', 'Database schema version'),
    ('index_version', '0', 'FAISS index build version'),
    ('last_index_build', '', 'Timestamp of last successful index build'),
    ('total_chunks', '0', 'Total number of chunks in database'),
    ('embedding_model', 'all-MiniLM-L6-v2', 'Model used for embeddings');