# Ingestion framework initialization
from .base_loader import AbstractLoader, Document, Source
from .file_loader import FileLoader, create_file_source
from .web_loader import WebLoader, create_web_source, create_single_url_source
from .wiki_loader import WikiLoader, create_wiki_source, create_wiki_search_source
from .arxiv_loader import ArxivLoader, create_arxiv_source, create_arxiv_category_source
from .pipeline_ingest import IngestionPipeline, TextNormalizer, TextChunker

__all__ = [
    'AbstractLoader', 'Document', 'Source',
    'FileLoader', 'create_file_source',
    'WebLoader', 'create_web_source', 'create_single_url_source',
    'WikiLoader', 'create_wiki_source', 'create_wiki_search_source',
    'ArxivLoader', 'create_arxiv_source', 'create_arxiv_category_source',
    'IngestionPipeline', 'TextNormalizer', 'TextChunker'
]