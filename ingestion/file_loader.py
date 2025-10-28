"""
File loader for local documents

Handles .txt, .pdf, .docx files from local filesystem.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import logging

# Import existing text extraction functionality
from enhanced_pipeline import TextExtractor

from .base_loader import AbstractLoader, Document, Source
from config import SUPPORTED_EXTENSIONS, PDF_MAX_PAGES, DOCX_MAX_PARAGRAPHS

logger = logging.getLogger(__name__)


class FileLoader(AbstractLoader):
    """Loads documents from local files"""
    
    def __init__(self, source: Source):
        super().__init__(source)
        self.text_extractor = TextExtractor()
        
        # Parse source locator as directory path
        self.root_path = Path(source.locator)
        if not self.root_path.exists():
            raise ValueError(f"Source path does not exist: {source.locator}")
            
    def discover(self) -> List[str]:
        """Discover all supported files in the directory tree"""
        files = []
        
        if self.root_path.is_file():
            # Single file
            if self.root_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(str(self.root_path))
        else:
            # Directory - walk recursively
            for ext in SUPPORTED_EXTENSIONS:
                pattern = f"**/*{ext}"
                files.extend(str(f) for f in self.root_path.glob(pattern))
                
        self.logger.info(f"Discovered {len(files)} files in {self.root_path}")
        return files
        
    def load(self, file_path: str) -> Document:
        """Load a single file"""
        path = Path(file_path)
        
        try:
            # Extract text using existing functionality
            content = self.text_extractor.extract_text(str(path))
            
            # Basic validation
            if not content or len(content.strip()) < 50:
                raise ValueError("Document too short or empty")
                
            # Get file metadata
            stat = path.stat()
            metadata = {
                'file_size': stat.st_size,
                'file_extension': path.suffix.lower(),
                'file_mtime': stat.st_mtime,
                'relative_path': str(path.relative_to(self.root_path)) if path.is_relative_to(self.root_path) else str(path)
            }
            
            # Apply file type specific limits
            if path.suffix.lower() == '.pdf':
                content = self._limit_pdf_content(content)
            elif path.suffix.lower() == '.docx':
                content = self._limit_docx_content(content)
                
            doc = Document(
                title=path.stem,
                content=content,
                source_locator=str(path),
                metadata=metadata
            )
            
            return doc
            
        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {e}")
            raise
            
    def _limit_pdf_content(self, content: str) -> str:
        """Apply PDF-specific content limits"""
        # Rough approximation: 500 chars per page
        max_chars = PDF_MAX_PAGES * 500
        if len(content) > max_chars:
            self.logger.warning(f"PDF content truncated from {len(content)} to {max_chars} characters")
            return content[:max_chars] + "\n[Content truncated...]"
        return content
        
    def _limit_docx_content(self, content: str) -> str:
        """Apply DOCX-specific content limits"""
        # Split by paragraphs and limit
        paragraphs = content.split('\n\n')
        if len(paragraphs) > DOCX_MAX_PARAGRAPHS:
            self.logger.warning(f"DOCX content truncated from {len(paragraphs)} to {DOCX_MAX_PARAGRAPHS} paragraphs")
            return '\n\n'.join(paragraphs[:DOCX_MAX_PARAGRAPHS]) + "\n[Content truncated...]"
        return content


def create_file_source(path: str, **metadata) -> Source:
    """Create a file source"""
    return Source(
        type='file',
        locator=str(Path(path).resolve()),
        metadata=metadata
    )