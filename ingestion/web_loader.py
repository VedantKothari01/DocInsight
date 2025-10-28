"""
Web loader for URL-based content

Fetches and processes web pages with respect for robots.txt.
"""

import requests
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import time
from typing import List, Dict, Any, Optional
import logging

from .base_loader import AbstractLoader, Document, Source
from config import MAX_WEB_CONCURRENCY

logger = logging.getLogger(__name__)

# Optional dependency
try:
    from readability import Document as ReadabilityDocument
    HAS_READABILITY = True
except ImportError:
    try:
        from newspaper import Article
        HAS_NEWSPAPER = True
    except ImportError:
        HAS_NEWSPAPER = False
    HAS_READABILITY = False


class WebLoader(AbstractLoader):
    """Loads documents from web URLs"""
    
    def __init__(self, source: Source):
        super().__init__(source)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DocInsight/2.0 (Document Analysis; +https://github.com/vedantkothari01/docinsight)'
        })
        
        # Parse URLs from source locator
        if isinstance(source.locator, str):
            # Single URL or newline-separated URLs
            self.urls = [url.strip() for url in source.locator.split('\n') if url.strip()]
        elif isinstance(source.locator, list):
            self.urls = source.locator
        else:
            raise ValueError("Source locator must be URL string or list of URLs")
            
        self.robots_cache = {}  # Cache robots.txt checks
        
    def discover(self) -> List[str]:
        """Return the list of URLs to process"""
        valid_urls = []
        
        for url in self.urls:
            if self._check_robots_txt(url):
                valid_urls.append(url)
            else:
                self.logger.warning(f"Robots.txt disallows: {url}")
                
        return valid_urls
        
    def load(self, url: str) -> Document:
        """Load content from a single URL"""
        try:
            # Fetch the page
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Extract readable content
            content = self._extract_content(response.text, url)
            
            if not content or len(content.strip()) < 100:
                raise ValueError("Insufficient content extracted")
                
            # Extract title
            title = self._extract_title(response.text, url)
            
            metadata = {
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type', ''),
                'content_length': len(response.text),
                'final_url': response.url,
                'extraction_method': self._get_extraction_method()
            }
            
            doc = Document(
                title=title,
                content=content,
                source_locator=url,
                url=response.url,
                metadata=metadata
            )
            
            # Be nice to servers
            time.sleep(0.5)
            
            return doc
            
        except Exception as e:
            self.logger.error(f"Failed to load {url}: {e}")
            raise
            
    def _check_robots_txt(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt"""
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        
        if robots_url not in self.robots_cache:
            try:
                rp = RobotFileParser()
                rp.set_url(robots_url)
                rp.read()
                self.robots_cache[robots_url] = rp
            except Exception:
                # If we can't read robots.txt, assume it's okay
                self.robots_cache[robots_url] = None
                
        rp = self.robots_cache[robots_url]
        if rp is None:
            return True
            
        return rp.can_fetch('*', url)
        
    def _extract_content(self, html: str, url: str) -> str:
        """Extract readable content from HTML"""
        if HAS_READABILITY:
            try:
                doc = ReadabilityDocument(html)
                return doc.summary()
            except Exception as e:
                self.logger.debug(f"Readability failed for {url}: {e}")
                
        if HAS_NEWSPAPER:
            try:
                article = Article(url)
                article.set_html(html)
                article.parse()
                return article.text
            except Exception as e:
                self.logger.debug(f"Newspaper failed for {url}: {e}")
                
        # Fallback: basic HTML stripping
        return self._basic_html_strip(html)
        
    def _extract_title(self, html: str, url: str) -> str:
        """Extract title from HTML"""
        try:
            import re
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
            if title_match:
                return title_match.group(1).strip()
        except Exception:
            pass
            
        # Fallback to URL path
        parsed = urlparse(url)
        return parsed.path.split('/')[-1] or parsed.netloc
        
    def _basic_html_strip(self, html: str) -> str:
        """Basic HTML tag removal as fallback"""
        import re
        # Remove script and style elements
        html = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        html = re.sub(r'<[^>]+>', '', html)
        # Clean up whitespace
        html = re.sub(r'\s+', ' ', html)
        return html.strip()
        
    def _get_extraction_method(self) -> str:
        """Get the extraction method used"""
        if HAS_READABILITY:
            return 'readability'
        elif HAS_NEWSPAPER:
            return 'newspaper'
        else:
            return 'basic_strip'


def create_web_source(urls: List[str], **metadata) -> Source:
    """Create a web source from list of URLs"""
    return Source(
        type='web',
        locator=urls,
        metadata=metadata
    )


def create_single_url_source(url: str, **metadata) -> Source:
    """Create a web source from single URL"""
    return Source(
        type='web',
        locator=url,
        metadata=metadata
    )