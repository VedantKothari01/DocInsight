"""
Base loader abstract class for DocInsight ingestion system

Defines the contract for all ingestion loaders.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document discovered by a loader"""
    title: str
    content: str
    source_locator: str  # Original path/URL
    url: Optional[str] = None  # Canonical URL if applicable
    metadata: Optional[Dict[str, Any]] = None
    language: str = 'en'
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Source:
    """Represents an ingestion source"""
    type: str  # 'file', 'web', 'wiki', 'arxiv'
    locator: str  # Path, URL, query, etc.
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AbstractLoader(ABC):
    """Abstract base class for all document loaders"""
    
    def __init__(self, source: Source):
        self.source = source
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    def discover(self) -> List[str]:
        """
        Discover available documents from the source
        
        Returns:
            List of identifiers/paths that can be loaded
        """
        pass
        
    @abstractmethod
    def load(self, identifier: str) -> Document:
        """
        Load a specific document by identifier
        
        Args:
            identifier: Document identifier from discover()
            
        Returns:
            Document object with content and metadata
        """
        pass
        
    def load_all(self) -> Iterator[Document]:
        """
        Load all documents from the source
        
        Yields:
            Document objects
        """
        try:
            identifiers = self.discover()
            self.logger.info(f"Discovered {len(identifiers)} documents from {self.source.locator}")
            
            for identifier in identifiers:
                try:
                    doc = self.load(identifier)
                    if doc and doc.content.strip():
                        yield doc
                    else:
                        self.logger.warning(f"Empty document: {identifier}")
                except Exception as e:
                    self.logger.error(f"Failed to load {identifier}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Failed to discover documents from {self.source.locator}: {e}")
            
    def validate_document(self, doc: Document) -> bool:
        """
        Validate a document before processing
        
        Args:
            doc: Document to validate
            
        Returns:
            True if document is valid
        """
        if not doc.content or not doc.content.strip():
            return False
            
        if len(doc.content) < 50:  # Minimum content length
            return False
            
        return True
        
    def get_source_metadata(self) -> Dict[str, Any]:
        """Get metadata about this source"""
        return {
            'type': self.source.type,
            'locator': self.source.locator,
            'loader_class': self.__class__.__name__,
            **self.source.metadata
        }