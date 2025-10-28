"""
Wikipedia loader using Wikipedia API

Fetches articles and content from Wikipedia.
"""

import requests
import time
from typing import List, Dict, Any, Optional
import logging

from .base_loader import AbstractLoader, Document, Source
from config import WIKI_MAX_PAGES

logger = logging.getLogger(__name__)


class WikiLoader(AbstractLoader):
    """Loads documents from Wikipedia"""
    
    def __init__(self, source: Source):
        super().__init__(source)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DocInsight/2.0 (Document Analysis; +https://github.com/vedantkothari01/docinsight)'
        })
        
        self.api_url = "https://en.wikipedia.org/api/rest_v1"
        self.search_url = "https://en.wikipedia.org/w/api.php"
        
        # Parse search terms or page titles from source locator
        if isinstance(source.locator, str):
            self.search_terms = [term.strip() for term in source.locator.split(',') if term.strip()]
        elif isinstance(source.locator, list):
            self.search_terms = source.locator
        else:
            raise ValueError("Source locator must be search terms string or list")
            
    def discover(self) -> List[str]:
        """Discover Wikipedia pages based on search terms"""
        pages = []
        
        for search_term in self.search_terms:
            try:
                # Search for pages
                search_results = self._search_pages(search_term)
                pages.extend(search_results)
                
                # Respect rate limits
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Search failed for '{search_term}': {e}")
                continue
                
        # Remove duplicates and limit results
        unique_pages = list(dict.fromkeys(pages))  # Preserves order
        if len(unique_pages) > WIKI_MAX_PAGES:
            self.logger.warning(f"Limiting results to {WIKI_MAX_PAGES} pages (found {len(unique_pages)})")
            unique_pages = unique_pages[:WIKI_MAX_PAGES]
            
        return unique_pages
        
    def load(self, page_title: str) -> Document:
        """Load a Wikipedia page"""
        try:
            # Get page content
            content = self._get_page_content(page_title)
            
            if not content or len(content.strip()) < 200:
                raise ValueError("Page content too short or empty")
                
            # Get page info
            page_info = self._get_page_info(page_title)
            
            metadata = {
                'wikipedia_url': f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}",
                'page_id': page_info.get('id'),
                'last_modified': page_info.get('timestamp'),
                'content_model': page_info.get('contentmodel'),
                'extract_method': 'wikipedia_api'
            }
            
            doc = Document(
                title=page_title,
                content=content,
                source_locator=page_title,
                url=metadata['wikipedia_url'],
                metadata=metadata
            )
            
            # Be nice to Wikipedia's servers
            time.sleep(0.1)
            
            return doc
            
        except Exception as e:
            self.logger.error(f"Failed to load Wikipedia page '{page_title}': {e}")
            raise
            
    def _search_pages(self, search_term: str, max_results: int = 10) -> List[str]:
        """Search for Wikipedia pages"""
        params = {
            'action': 'opensearch',
            'search': search_term,
            'limit': max_results,
            'namespace': 0,  # Main namespace only
            'format': 'json'
        }
        
        response = self.session.get(self.search_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if len(data) >= 2:
            return data[1]  # Page titles are in the second element
        return []
        
    def _get_page_content(self, page_title: str) -> str:
        """Get the full content of a Wikipedia page"""
        # First try to get the page content via API
        url = f"{self.api_url}/page/summary/{page_title.replace(' ', '_')}"
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # Get the extract (summary)
            extract = data.get('extract', '')
            
            # Try to get more detailed content
            full_content = self._get_full_page_content(page_title)
            if full_content:
                return full_content
            else:
                return extract
                
        except Exception as e:
            self.logger.debug(f"API summary failed for {page_title}: {e}")
            # Fallback to full content
            return self._get_full_page_content(page_title) or ""
            
    def _get_full_page_content(self, page_title: str) -> Optional[str]:
        """Get full page wikitext content"""
        params = {
            'action': 'query',
            'format': 'json',
            'titles': page_title,
            'prop': 'extracts',
            'explaintext': True,
            'exsectionformat': 'plain'
        }
        
        try:
            response = self.session.get(self.search_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                extract = page_data.get('extract', '')
                if extract:
                    return extract
                    
        except Exception as e:
            self.logger.debug(f"Full content fetch failed for {page_title}: {e}")
            
        return None
        
    def _get_page_info(self, page_title: str) -> Dict[str, Any]:
        """Get page metadata"""
        params = {
            'action': 'query',
            'format': 'json',
            'titles': page_title,
            'prop': 'info'
        }
        
        try:
            response = self.session.get(self.search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                return page_data
                
        except Exception as e:
            self.logger.debug(f"Page info fetch failed for {page_title}: {e}")
            
        return {}


def create_wiki_source(search_terms: List[str], **metadata) -> Source:
    """Create a Wikipedia source from search terms"""
    return Source(
        type='wiki',
        locator=search_terms,
        metadata=metadata
    )


def create_wiki_search_source(search_query: str, **metadata) -> Source:
    """Create a Wikipedia source from search query string"""
    return Source(
        type='wiki',
        locator=search_query,
        metadata=metadata
    )