"""
arXiv loader for academic papers

Fetches abstracts and metadata from arXiv.org.
"""

import requests
import xml.etree.ElementTree as ET
import time
from typing import List, Dict, Any, Optional
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import logging

from .base_loader import AbstractLoader, Document, Source
from enhanced_pipeline import TextExtractor
from config import ARXIV_RATE_LIMIT

logger = logging.getLogger(__name__)


class ArxivLoader(AbstractLoader):
    """Loads documents from arXiv.org"""
    
    def __init__(self, source: Source):
        super().__init__(source)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DocInsight/2.0 (Document Analysis; +https://github.com/vedantkothari01/docinsight)'
        })
        
        self.api_url = "http://export.arxiv.org/api/query"
        self.rate_limit = ARXIV_RATE_LIMIT
        self.text_extractor = TextExtractor()
        
        # Parse search parameters from source locator
        self.search_params = self._parse_search_params(source.locator)
        
    def _parse_search_params(self, locator: str) -> Dict[str, Any]:
        """Parse search parameters from locator string"""
        params = {
            'search_query': 'all:machine learning',  # Default
            'max_results': 50,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        if isinstance(locator, dict):
            params.update(locator)
        elif isinstance(locator, str):
            # Simple search query
            params['search_query'] = locator
        
        return params
        
    def discover(self) -> List[str]:
        """Discover arXiv papers based on search query"""
        try:
            # Build API request
            params = {
                'search_query': self.search_params['search_query'],
                'max_results': self.search_params['max_results'],
                'sortBy': self.search_params['sortBy'],
                'sortOrder': self.search_params['sortOrder']
            }
            
            response = self.session.get(self.api_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Extract paper IDs
            paper_ids = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                id_elem = entry.find('{http://www.w3.org/2005/Atom}id')
                if id_elem is not None:
                    # Extract ID from URL like "http://arxiv.org/abs/2301.12345v1"
                    arxiv_url = id_elem.text
                    if '/abs/' in arxiv_url:
                        arxiv_id = arxiv_url.split('/abs/')[-1]
                        # Remove version suffix if present (e.g., "v1", "v2")
                        if '.' in arxiv_id and arxiv_id.split('.')[-1].startswith('v'):
                            arxiv_id = '.'.join(arxiv_id.split('.')[:-1])
                        paper_ids.append(arxiv_id)
                    
            self.logger.info(f"Discovered {len(paper_ids)} arXiv papers")
            return paper_ids
            
        except Exception as e:
            self.logger.error(f"arXiv discovery failed: {e}")
            return []
            
    def load(self, arxiv_id: str) -> Document:
        """Load a single arXiv paper"""
        try:
            # Get paper details
            params = {
                'id_list': arxiv_id,
                'max_results': 1
            }
            
            response = self.session.get(self.api_url, params=params, timeout=15)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            entry = root.find('{http://www.w3.org/2005/Atom}entry')
            
            if entry is None:
                raise ValueError(f"Paper not found: {arxiv_id}")
                
            # Extract paper data
            paper_data = self._extract_paper_data(entry)
            
            # Prefer full-text PDF content when available; fall back to abstract
            content = None
            try:
                content = self._try_fetch_fulltext_pdf(arxiv_id)
            except Exception as pdf_err:
                self.logger.warning(f"PDF fetch failed for {arxiv_id}: {pdf_err}. Falling back to abstract.")
                content = None

            if not content:
                # Create document from metadata + abstract
                content = self._build_content(paper_data)
            
            if len(content.strip()) < 200:
                raise ValueError("Content too short")
                
            metadata = {
                'arxiv_id': arxiv_id,
                'arxiv_url': f"https://arxiv.org/abs/{arxiv_id}",
                'authors': paper_data.get('authors', []),
                'categories': paper_data.get('categories', []),
                'published': paper_data.get('published'),
                'updated': paper_data.get('updated'),
                'doi': paper_data.get('doi'),
                'journal_ref': paper_data.get('journal_ref'),
                'comment': paper_data.get('comment')
            }
            
            doc = Document(
                title=paper_data.get('title', f"arXiv:{arxiv_id}"),
                content=content,
                source_locator=arxiv_id,
                url=metadata['arxiv_url'],
                metadata=metadata
            )
            
            # Respect rate limits
            time.sleep(self.rate_limit)
            
            return doc
            
        except Exception as e:
            self.logger.error(f"Failed to load arXiv paper {arxiv_id}: {e}")
            raise

    def _try_fetch_fulltext_pdf(self, arxiv_id: str) -> Optional[str]:
        """Attempt to download and extract full-text PDF for the arXiv paper.

        Returns extracted text or None if unavailable.
        """
        # Normalize id: arXiv API sometimes returns IDs without version; PDFs accept both
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / f"{arxiv_id}.pdf"
            resp = self.session.get(pdf_url, timeout=60)
            if resp.status_code != 200 or not resp.content:
                raise ValueError(f"PDF not accessible ({resp.status_code})")
            pdf_path.write_bytes(resp.content)
            # Extract text via shared extractor (handles PDF limits via config)
            text = self.text_extractor.extract_text(str(pdf_path))
            # Basic sanity: must be substantial text
            if not text or len(text.strip()) < 2000:
                raise ValueError("Extracted PDF text too short")
            return text
            
    def _extract_paper_data(self, entry) -> Dict[str, Any]:
        """Extract data from arXiv API entry"""
        data = {}
        
        # Title
        title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
        if title_elem is not None:
            data['title'] = title_elem.text.strip()
            
        # Abstract
        summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
        if summary_elem is not None:
            data['abstract'] = summary_elem.text.strip()
            
        # Authors
        authors = []
        for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
            name_elem = author.find('{http://www.w3.org/2005/Atom}name')
            if name_elem is not None:
                authors.append(name_elem.text)
        data['authors'] = authors
        
        # Categories
        categories = []
        for category in entry.findall('{http://arxiv.org/schemas/atom}primary_category'):
            term = category.get('term')
            if term:
                categories.append(term)
        data['categories'] = categories
        
        # Dates
        published_elem = entry.find('{http://www.w3.org/2005/Atom}published')
        if published_elem is not None:
            data['published'] = published_elem.text
            
        updated_elem = entry.find('{http://www.w3.org/2005/Atom}updated')
        if updated_elem is not None:
            data['updated'] = updated_elem.text
            
        # DOI
        for link in entry.findall('{http://www.w3.org/2005/Atom}link'):
            if link.get('title') == 'doi':
                data['doi'] = link.get('href')
                
        # Journal reference
        journal_elem = entry.find('{http://arxiv.org/schemas/atom}journal_ref')
        if journal_elem is not None:
            data['journal_ref'] = journal_elem.text
            
        # Comment
        comment_elem = entry.find('{http://arxiv.org/schemas/atom}comment')
        if comment_elem is not None:
            data['comment'] = comment_elem.text
            
        return data
        
    def _build_content(self, paper_data: Dict[str, Any]) -> str:
        """Build document content from paper data"""
        parts = []
        
        # Title
        if 'title' in paper_data:
            parts.append(f"Title: {paper_data['title']}")
            
        # Authors
        if paper_data.get('authors'):
            authors_str = ', '.join(paper_data['authors'])
            parts.append(f"Authors: {authors_str}")
            
        # Abstract
        if 'abstract' in paper_data:
            parts.append(f"Abstract: {paper_data['abstract']}")
            
        # Categories
        if paper_data.get('categories'):
            cats_str = ', '.join(paper_data['categories'])
            parts.append(f"Categories: {cats_str}")
            
        # Journal reference
        if paper_data.get('journal_ref'):
            parts.append(f"Journal: {paper_data['journal_ref']}")
            
        # Comment
        if paper_data.get('comment'):
            parts.append(f"Comment: {paper_data['comment']}")
            
        return '\n\n'.join(parts)


def create_arxiv_source(search_query: str, max_results: int = 50, **metadata) -> Source:
    """Create an arXiv source from search query"""
    locator = {
        'search_query': search_query,
        'max_results': max_results,
        'sortBy': 'relevance',
        'sortOrder': 'descending'
    }
    
    return Source(
        type='arxiv',
        locator=locator,
        metadata=metadata
    )


def create_arxiv_category_source(category: str, max_results: int = 50, **metadata) -> Source:
    """Create an arXiv source for a specific category"""
    search_query = f"cat:{category}"
    return create_arxiv_source(search_query, max_results, **metadata)