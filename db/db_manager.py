"""
Database manager for DocInsight Phase 2

Handles SQLite database connections, migrations, and transactional operations.
"""

import sqlite3
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager
import json
import hashlib
from datetime import datetime

from config import DB_PATH

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database connections and operations"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        self._ensure_db_directory()
        self._connection = None
        
    def _ensure_db_directory(self):
        """Ensure database directory exists"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
    def connect(self) -> sqlite3.Connection:
        """Get database connection with proper configuration"""
        if self._connection is None:
            self._connection = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                check_same_thread=False
            )
            self._connection.row_factory = sqlite3.Row
            # Enable foreign keys
            self._connection.execute("PRAGMA foreign_keys = ON")
            
        return self._connection
        
    def close(self):
        """Close database connection"""
        if self._connection:
            self._connection.close()
            self._connection = None
            
    @contextmanager
    def transaction(self):
        """Context manager for database transactions"""
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
            
    def initialize_schema(self):
        """Initialize database schema from schema.sql"""
        schema_path = Path(__file__).parent / 'schema.sql'
        
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
            
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
            
        with self.transaction() as conn:
            conn.executescript(schema_sql)
            logger.info(f"Database schema initialized: {self.db_path}")
            
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get setting value from database"""
        conn = self.connect()
        cursor = conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        )
        result = cursor.fetchone()
        
        if result:
            try:
                # Try to parse as JSON first
                return json.loads(result['value'])
            except (json.JSONDecodeError, TypeError):
                # Return as string if not JSON
                return result['value']
        
        return default
        
    def set_setting(self, key: str, value: Any, description: str = None):
        """Set setting value in database"""
        # Convert value to JSON string if not already string
        if isinstance(value, str):
            value_str = value
        else:
            value_str = json.dumps(value)
            
        with self.transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO settings (key, value, description, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (key, value_str, description))
            
    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get corpus statistics"""
        conn = self.connect()
        
        stats = {}
        
        # Total documents
        cursor = conn.execute("SELECT COUNT(*) as count FROM documents")
        stats['total_documents'] = cursor.fetchone()['count']
        
        # Total chunks
        cursor = conn.execute("SELECT COUNT(*) as count FROM chunks")
        stats['total_chunks'] = cursor.fetchone()['count']
        
        # Embedded chunks
        cursor = conn.execute("SELECT COUNT(*) as count FROM chunks WHERE embedding IS NOT NULL")
        stats['embedded_chunks'] = cursor.fetchone()['count']
        
        # Sources by type
        cursor = conn.execute("""
            SELECT type, COUNT(*) as count 
            FROM sources 
            GROUP BY type
        """)
        stats['sources_by_type'] = {row['type']: row['count'] for row in cursor.fetchall()}
        
        # Total characters
        cursor = conn.execute("SELECT SUM(char_count) as total FROM documents")
        result = cursor.fetchone()
        stats['total_characters'] = result['total'] or 0
        
        return stats
        
    def purge_source(self, source_id: int):
        """Remove a source and all its documents/chunks"""
        with self.transaction() as conn:
            # Delete will cascade to documents and chunks
            conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
            logger.info(f"Purged source {source_id}")
            
    def purge_all(self):
        """Remove all data (dangerous!)"""
        with self.transaction() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents") 
            conn.execute("DELETE FROM sources")
            conn.execute("DELETE FROM ingestion_runs")
            logger.warning("Purged all corpus data")
            
    def vacuum(self):
        """Vacuum database to reclaim space"""
        conn = self.connect()
        conn.execute("VACUUM")
        logger.info("Database vacuumed")


class IngestionRun:
    """Context manager for tracking ingestion runs"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.run_id = None
        self.source_count = 0
        self.doc_count = 0
        self.chunk_count = 0
        
    def __enter__(self):
        with self.db_manager.transaction() as conn:
            cursor = conn.execute("""
                INSERT INTO ingestion_runs (started_at, status)
                VALUES (CURRENT_TIMESTAMP, 'running')
            """)
            self.run_id = cursor.lastrowid
            
        logger.info(f"Started ingestion run {self.run_id}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        status = 'completed' if exc_type is None else 'failed'
        error_message = str(exc_val) if exc_val else None
        
        with self.db_manager.transaction() as conn:
            conn.execute("""
                UPDATE ingestion_runs 
                SET finished_at = CURRENT_TIMESTAMP,
                    source_count = ?,
                    doc_count = ?,
                    chunk_count = ?,
                    status = ?,
                    error_message = ?
                WHERE id = ?
            """, (self.source_count, self.doc_count, self.chunk_count, 
                  status, error_message, self.run_id))
                  
        logger.info(f"Finished ingestion run {self.run_id}: {status}")
        
    def add_stats(self, sources: int = 0, docs: int = 0, chunks: int = 0):
        """Add to run statistics"""
        self.source_count += sources
        self.doc_count += docs
        self.chunk_count += chunks


def get_content_hash(content: str) -> str:
    """Generate SHA256 hash of content for deduplication"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


# Global database manager instance
_db_manager = None

def get_db_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        # Initialize schema if database doesn't exist
        if not Path(_db_manager.db_path).exists():
            _db_manager.initialize_schema()
    return _db_manager