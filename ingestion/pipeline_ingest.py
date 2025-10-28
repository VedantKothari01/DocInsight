"""
Ingestion pipeline orchestrator

Coordinates document discovery, loading, normalization, chunking, and storage.
"""

import logging
import json
import hashlib
import re
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path

# Language detection (optional dependency)
try:
    from langdetect import detect, LangDetectError
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

# NLTK for text processing
from nltk.tokenize import sent_tokenize

from .base_loader import AbstractLoader, Document, Source
from .file_loader import FileLoader
from .web_loader import WebLoader  
from .wiki_loader import WikiLoader
from .arxiv_loader import ArxivLoader
from db import DatabaseManager, IngestionRun, get_content_hash
from config import (
    CHUNK_SIZE_TOKENS, CHUNK_OVERLAP, MIN_CHUNK_LENGTH, MAX_CHUNK_LENGTH,
    CHUNKING_STRATEGY, SUPPORTED_LANGUAGES, LANGUAGE_DETECTION_CONFIDENCE,
    MAX_CHUNKS_PER_DOC
)

logger = logging.getLogger(__name__)


class TextNormalizer:
    """Handles text normalization and preprocessing"""
    
    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text content"""
        if not text:
            return ""
            
        # Unicode normalization
        import unicodedata
        text = unicodedata.normalize('NFC', text)
        
        # Basic cleanup
        text = text.strip()
        
        # Collapse multiple whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines but preserve paragraph structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text
        
    @staticmethod
    def detect_language(text: str) -> Optional[str]:
        """Detect language of text"""
        if not HAS_LANGDETECT:
            return 'en'  # Default assumption
            
        try:
            # Use first 1000 characters for detection
            sample = text[:1000] if len(text) > 1000 else text
            lang = detect(sample)
            return lang
        except (LangDetectError, Exception):
            return 'en'  # Default fallback


class TextChunker:
    """Handles text chunking strategies"""
    
    def __init__(self, strategy: str = CHUNKING_STRATEGY):
        self.strategy = strategy
        
    def chunk_text(self, text: str, max_chunks: int = MAX_CHUNKS_PER_DOC) -> List[str]:
        """Split text into chunks"""
        if self.strategy == 'sentence':
            return self._sentence_chunking(text, max_chunks)
        elif self.strategy == 'sliding_window':
            return self._sliding_window_chunking(text, max_chunks)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")
            
    def _sentence_chunking(self, text: str, max_chunks: int) -> List[str]:
        """Chunk by sentences with target token count"""
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            logger.warning(f"Sentence tokenization failed: {e}")
            # Fallback to paragraph splitting
            sentences = [s.strip() for s in text.split('\n\n') if s.strip()]
            
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Estimate tokens (rough approximation: 4 chars per token)
            estimated_tokens = current_length // 4
            sentence_tokens = sentence_length // 4
            
            if estimated_tokens + sentence_tokens > CHUNK_SIZE_TOKENS and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= MIN_CHUNK_LENGTH:
                    chunks.append(chunk_text)
                    
                # Start new chunk with overlap
                if len(current_chunk) > 1:
                    overlap_sentences = current_chunk[-1:]  # Keep last sentence
                    current_chunk = overlap_sentences + [sentence]
                    current_length = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
                
            # Stop if we have enough chunks
            if len(chunks) >= max_chunks:
                break
                
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= MIN_CHUNK_LENGTH:
                chunks.append(chunk_text)
                
        return chunks[:max_chunks]
        
    def _sliding_window_chunking(self, text: str, max_chunks: int) -> List[str]:
        """Sliding window chunking with character-based windows"""
        chunks = []
        chunk_size = min(CHUNK_SIZE_TOKENS * 4, MAX_CHUNK_LENGTH)  # Rough char estimate
        overlap_size = CHUNK_OVERLAP * 4
        
        start = 0
        while start < len(text) and len(chunks) < max_chunks:
            end = start + chunk_size
            
            # Try to end at word boundary
            if end < len(text):
                # Look back for word boundary
                for i in range(min(50, end - start)):
                    if text[end - i].isspace():
                        end = end - i
                        break
                        
            chunk = text[start:end].strip()
            
            if len(chunk) >= MIN_CHUNK_LENGTH:
                chunks.append(chunk)
                
            start = end - overlap_size
            if start >= len(text) - MIN_CHUNK_LENGTH:
                break
                
        return chunks


class IngestionPipeline:
    """Main ingestion pipeline orchestrator"""
    
    def __init__(self, db_manager: DatabaseManager = None):
        self.db_manager = db_manager or DatabaseManager()
        self.normalizer = TextNormalizer()
        self.chunker = TextChunker()
        
        # Loader registry
        self.loaders = {
            'file': FileLoader,
            'web': WebLoader,
            'wiki': WikiLoader,
            'arxiv': ArxivLoader
        }
        
    def register_loader(self, source_type: str, loader_class):
        """Register a custom loader"""
        self.loaders[source_type] = loader_class
        
    def ingest_source(self, source: Source) -> Dict[str, Any]:
        """Ingest documents from a source"""
        if source.type not in self.loaders:
            raise ValueError(f"Unknown source type: {source.type}")
            
        loader_class = self.loaders[source.type]
        loader = loader_class(source)
        
        stats = {
            'source_type': source.type,
            'source_locator': source.locator,
            'documents_processed': 0,
            'chunks_created': 0,
            'documents_skipped': 0,
            'errors': []
        }
        
        try:
            with IngestionRun(self.db_manager) as run:
                # Store source in database
                source_id = self._store_source(source)
                run.add_stats(sources=1)
                
                # Process documents
                for doc in loader.load_all():
                    try:
                        if self._process_document(doc, source_id):
                            stats['documents_processed'] += 1
                            # Count chunks for this document
                            chunk_count = self._count_document_chunks(source_id, doc)
                            stats['chunks_created'] += chunk_count
                            run.add_stats(docs=1, chunks=chunk_count)
                        else:
                            stats['documents_skipped'] += 1
                    except Exception as e:
                        error_msg = f"Document processing failed: {e}"
                        stats['errors'].append(error_msg)
                        logger.error(error_msg)
                        continue
                        
        except Exception as e:
            stats['errors'].append(f"Ingestion failed: {e}")
            logger.error(f"Ingestion failed for {source.locator}: {e}")
            
        return stats
        
    def _store_source(self, source: Source) -> int:
        """Store source in database"""
        meta_json = json.dumps(source.metadata) if source.metadata else None
        
        with self.db_manager.transaction() as conn:
            cursor = conn.execute("""
                INSERT INTO sources (type, locator, meta_json)
                VALUES (?, ?, ?)
            """, (source.type, str(source.locator), meta_json))
            return cursor.lastrowid
            
    def _process_document(self, doc: Document, source_id: int) -> bool:
        """Process a single document"""
        # Normalize content
        normalized_content = self.normalizer.normalize(doc.content)
        
        if len(normalized_content) < MIN_CHUNK_LENGTH:
            logger.debug(f"Document too short after normalization: {doc.title}")
            return False
            
        # Detect language
        language = self.normalizer.detect_language(normalized_content)
        if language not in SUPPORTED_LANGUAGES:
            logger.debug(f"Unsupported language '{language}' for document: {doc.title}")
            return False
            
        # Check for duplicate content
        content_hash = get_content_hash(normalized_content)
        if self._document_exists(content_hash):
            logger.debug(f"Duplicate document skipped: {doc.title}")
            return False
            
        # Store document
        doc_id = self._store_document(doc, source_id, content_hash, language, len(normalized_content))
        
        # Chunk and store
        chunks = self.chunker.chunk_text(normalized_content)
        self._store_chunks(doc_id, chunks)
        
        logger.info(f"Processed document '{doc.title}': {len(chunks)} chunks")
        return True
        
    def _document_exists(self, content_hash: str) -> bool:
        """Check if document with this content hash already exists"""
        conn = self.db_manager.connect()
        cursor = conn.execute(
            "SELECT 1 FROM documents WHERE content_hash = ? LIMIT 1",
            (content_hash,)
        )
        return cursor.fetchone() is not None
        
    def _store_document(self, doc: Document, source_id: int, content_hash: str, 
                       language: str, char_count: int) -> int:
        """Store document in database"""
        meta_json = json.dumps(doc.metadata) if doc.metadata else None
        
        with self.db_manager.transaction() as conn:
            cursor = conn.execute("""
                INSERT INTO documents 
                (source_id, title, raw_path, url, content_hash, char_count, language)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (source_id, doc.title, doc.source_locator, doc.url, 
                  content_hash, char_count, language))
            return cursor.lastrowid
            
    def _store_chunks(self, doc_id: int, chunks: List[str]):
        """Store chunks in database"""
        with self.db_manager.transaction() as conn:
            for ordinal, chunk_text in enumerate(chunks):
                chunk_hash = get_content_hash(chunk_text)
                token_count = len(chunk_text) // 4  # Rough estimate
                
                conn.execute("""
                    INSERT INTO chunks 
                    (document_id, ordinal, text, token_count, char_count, hash)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (doc_id, ordinal, chunk_text, token_count, len(chunk_text), chunk_hash))
                
    def _count_document_chunks(self, source_id: int, doc: Document) -> int:
        """Count chunks for the most recently added document"""
        conn = self.db_manager.connect()
        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.source_id = ? AND d.title = ?
            ORDER BY d.id DESC LIMIT 1
        """, (source_id, doc.title))
        result = cursor.fetchone()
        return result['count'] if result else 0