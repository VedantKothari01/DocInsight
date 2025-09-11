#!/usr/bin/env python3
"""
DocInsight CLI - Command Line Interface for Phase 2

Provides commands for corpus management, ingestion, indexing, and querying.
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Phase 2 components
try:
    from db import DatabaseManager, get_db_manager
    from ingestion import (
        IngestionPipeline, create_file_source, create_web_source,
        create_wiki_source, create_arxiv_source
    )
    from index import IndexManager
    from retrieval import RetrievalEngine
    from embeddings import EmbeddingProcessor, Embedder
except ImportError as e:
    logger.error(f"Failed to import DocInsight components: {e}")
    sys.exit(1)


class DocInsightCLI:
    """Main CLI class"""
    
    def __init__(self):
        self.db_manager = get_db_manager()
        self.ingestion_pipeline = IngestionPipeline(self.db_manager)
        self.index_manager = IndexManager(self.db_manager)
        self.retrieval_engine = RetrievalEngine(self.db_manager, self.index_manager)
        
    def cmd_ingest(self, args):
        """Ingest documents from a source"""
        print(f"🔄 Ingesting from {args.source_type}: {args.source}")
        
        try:
            # Create source based on type
            if args.source_type == 'file':
                source = create_file_source(args.source)
            elif args.source_type == 'web':
                # Expect comma-separated URLs
                urls = [url.strip() for url in args.source.split(',')]
                from ingestion import create_web_source
                source = create_web_source(urls)
            elif args.source_type == 'wiki':
                from ingestion import create_wiki_search_source
                source = create_wiki_search_source(args.source)
            elif args.source_type == 'arxiv':
                max_results = args.max_results or 50
                source = create_arxiv_source(args.source, max_results)
            else:
                print(f"❌ Unknown source type: {args.source_type}")
                return
                
            # Run ingestion
            stats = self.ingestion_pipeline.ingest_source(source)
            
            # Print results
            print(f"✅ Ingestion completed:")
            print(f"   📄 Documents processed: {stats['documents_processed']}")
            print(f"   📝 Chunks created: {stats['chunks_created']}")
            print(f"   ⚠️  Documents skipped: {stats['documents_skipped']}")
            
            if stats['errors']:
                print(f"   ❌ Errors: {len(stats['errors'])}")
                for error in stats['errors'][:3]:  # Show first 3 errors
                    print(f"      • {error}")
                    
        except Exception as e:
            print(f"❌ Ingestion failed: {e}")
            logger.error(f"Ingestion failed: {e}")
            
    def cmd_reindex(self, args):
        """Rebuild the search index"""
        print("🔄 Rebuilding search index...")
        
        try:
            stats = self.index_manager.build_index(force_rebuild=True)
            
            if stats['build_successful']:
                print(f"✅ Index rebuilt successfully:")
                print(f"   📊 Chunks indexed: {stats['chunks_indexed']}")
                print(f"   🧠 Embeddings generated: {stats['embeddings_generated']}")
                print(f"   📈 Index type: {stats['index_type']}")
            else:
                print(f"❌ Index rebuild failed:")
                for error in stats['errors']:
                    print(f"   • {error}")
                    
        except Exception as e:
            print(f"❌ Index rebuild failed: {e}")
            logger.error(f"Index rebuild failed: {e}")
            
    def cmd_stats(self, args):
        """Show corpus and index statistics"""
        print("📊 DocInsight Corpus Statistics")
        print("=" * 40)
        
        try:
            # Get comprehensive stats
            stats = self.retrieval_engine.get_corpus_stats()
            
            # Database stats
            print(f"📚 Corpus Overview:")
            print(f"   Documents: {stats.get('total_documents', 0):,}")
            print(f"   Text chunks: {stats.get('total_chunks', 0):,}")
            print(f"   Total characters: {stats.get('total_characters', 0):,}")
            print(f"   Embedded chunks: {stats.get('embedded_chunks', 0):,}")
            
            # Sources breakdown
            sources = stats.get('sources_by_type', {})
            if sources:
                print(f"\n📁 Sources by type:")
                for source_type, count in sources.items():
                    print(f"   {source_type}: {count}")
                    
            # Index stats
            print(f"\n🔍 Search Index:")
            print(f"   Status: {'✅ Ready' if stats.get('available', False) else '❌ Not available'}")
            print(f"   Type: {stats.get('index_type', 'Unknown')}")
            print(f"   Indexed vectors: {stats.get('num_vectors', 0):,}")
            print(f"   Dimension: {stats.get('dimension', 'Unknown')}")
            
            coverage = stats.get('index_coverage', 0)
            print(f"   Coverage: {coverage:.1%}")
            
            if coverage < 1.0 and stats.get('embedded_chunks', 0) > 0:
                print("   ⚠️  Index may need rebuilding")
                
        except Exception as e:
            print(f"❌ Failed to get statistics: {e}")
            logger.error(f"Stats failed: {e}")
            
    def cmd_search(self, args):
        """Search the corpus"""
        query = args.query
        top_k = args.top_k or 5
        
        print(f"🔍 Searching for: '{query}'")
        print(f"📊 Top {top_k} results:")
        print("-" * 80)
        
        try:
            if not self.retrieval_engine.is_ready():
                print("❌ Search index not ready. Run 'reindex' first.")
                return
                
            # Search
            results = self.retrieval_engine.retrieve_similar_chunks(
                [query], top_k=top_k
            )
            
            if not results:
                print("No results found.")
                return
                
            # Display results
            for i, result in enumerate(results, 1):
                print(f"\n{i}. 📄 {result.document_title}")
                print(f"   📍 {result.source_type}: {result.source_locator}")
                print(f"   📊 Score: {result.score:.3f}")
                print(f"   📝 Text: {result.text[:200]}...")
                if len(result.text) > 200:
                    print("   [truncated]")
                    
        except Exception as e:
            print(f"❌ Search failed: {e}")
            logger.error(f"Search failed: {e}")
            
    def cmd_purge(self, args):
        """Purge corpus data"""
        if args.source_id:
            print(f"🗑️  Purging source {args.source_id}...")
            try:
                self.db_manager.purge_source(args.source_id)
                print(f"✅ Source {args.source_id} purged")
            except Exception as e:
                print(f"❌ Failed to purge source: {e}")
                
        elif args.all:
            if not args.confirm:
                print("❌ Use --confirm to purge all data")
                return
                
            print("🗑️  Purging all corpus data...")
            try:
                self.db_manager.purge_all()
                self.index_manager.clear_index()
                print("✅ All data purged")
            except Exception as e:
                print(f"❌ Failed to purge data: {e}")
        else:
            print("❌ Specify --source-id or --all")
            
    def cmd_validate(self, args):
        """Validate corpus integrity"""
        print("🔍 Validating corpus integrity...")
        
        issues = []
        
        try:
            # Check database
            stats = self.db_manager.get_corpus_stats()
            
            # Check for orphaned chunks
            conn = self.db_manager.connect()
            cursor = conn.execute("""
                SELECT COUNT(*) as count FROM chunks c
                LEFT JOIN documents d ON c.document_id = d.id
                WHERE d.id IS NULL
            """)
            orphaned_chunks = cursor.fetchone()['count']
            
            if orphaned_chunks > 0:
                issues.append(f"Found {orphaned_chunks} orphaned chunks")
                
            # Check for documents without chunks
            cursor = conn.execute("""
                SELECT COUNT(*) as count FROM documents d
                LEFT JOIN chunks c ON d.id = c.document_id
                WHERE c.id IS NULL
            """)
            empty_docs = cursor.fetchone()['count']
            
            if empty_docs > 0:
                issues.append(f"Found {empty_docs} documents without chunks")
                
            # Check embedding consistency
            embedded_ratio = stats['embedded_chunks'] / max(stats['total_chunks'], 1)
            if embedded_ratio < 0.9:
                issues.append(f"Only {embedded_ratio:.1%} of chunks have embeddings")
                
            # Check index consistency
            index_stats = self.index_manager.get_index_stats()
            if index_stats.get('available') and index_stats.get('index_coverage', 0) < 0.9:
                issues.append("Search index may be outdated")
                
            # Report results
            if issues:
                print(f"⚠️  Found {len(issues)} issues:")
                for issue in issues:
                    print(f"   • {issue}")
                    
                print("\n💡 Suggested actions:")
                print("   • Run 'reindex' to rebuild search index")
                print("   • Check ingestion logs for errors")
            else:
                print("✅ Corpus validation passed - no issues found")
                
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            logger.error(f"Validation failed: {e}")
            
    def cmd_embed(self, args):
        """Generate embeddings for unembedded chunks"""
        print("🧠 Generating embeddings...")
        
        try:
            embedder = Embedder()
            if not embedder.is_available():
                print("❌ Embedding model not available")
                return
                
            processor = EmbeddingProcessor(embedder)
            stats = processor.process_unembedded_chunks(self.db_manager)
            
            print(f"✅ Embedding generation completed:")
            print(f"   📊 Chunks processed: {stats['chunks_processed']}")
            print(f"   ❌ Chunks failed: {stats['chunks_failed']}")
            print(f"   📦 Batches processed: {stats['batches_processed']}")
            
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            logger.error(f"Embedding generation failed: {e}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="DocInsight CLI - Document corpus management"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents')
    ingest_parser.add_argument('source_type', choices=['file', 'web', 'wiki', 'arxiv'],
                              help='Type of source to ingest')
    ingest_parser.add_argument('source', help='Source locator (path, URL, query, etc.)')
    ingest_parser.add_argument('--max-results', type=int, 
                              help='Maximum results for API sources')
    
    # Reindex command
    reindex_parser = subparsers.add_parser('reindex', help='Rebuild search index')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show corpus statistics')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search corpus')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--top-k', type=int, default=5,
                              help='Number of results to return')
    
    # Purge command
    purge_parser = subparsers.add_parser('purge', help='Purge corpus data')
    purge_group = purge_parser.add_mutually_exclusive_group(required=True)
    purge_group.add_argument('--source-id', type=int, help='Purge specific source')
    purge_group.add_argument('--all', action='store_true', help='Purge all data')
    purge_parser.add_argument('--confirm', action='store_true',
                             help='Confirm purge operation')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate corpus integrity')
    
    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Generate embeddings')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    # Initialize CLI
    try:
        cli = DocInsightCLI()
    except Exception as e:
        print(f"❌ Failed to initialize DocInsight: {e}")
        logger.error(f"CLI initialization failed: {e}")
        sys.exit(1)
        
    # Execute command
    command_method = getattr(cli, f'cmd_{args.command}', None)
    if command_method:
        command_method(args)
    else:
        print(f"❌ Unknown command: {args.command}")
        parser.print_help()


if __name__ == '__main__':
    main()