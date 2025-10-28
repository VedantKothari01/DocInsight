#!/usr/bin/env python3
import logging
from db import get_db_manager
from ingestion import IngestionPipeline, create_arxiv_category_source
from embeddings import EmbeddingProcessor
from index import IndexManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Target ~300 docs total
DOC_CAP = 300
ARXIV_CATEGORIES = [
    'cs.AI','cs.CL','cs.CV','cs.LG','cs.NE','cs.DS','cs.CR','cs.DB','cs.OS','cs.PL','cs.HC','cs.IT','cs.CC','cs.SE','cs.NI'
]
PER_CATEGORY = 10 

if __name__ == '__main__':
    db = get_db_manager()
    pipeline = IngestionPipeline(db)

    # Count existing arXiv docs
    conn = db.connect()
    existing = conn.execute("""
        SELECT COUNT(*) as c
        FROM documents d JOIN sources s ON d.source_id = s.id
        WHERE s.type='arxiv'
    """).fetchone()['c']
    logger.info(f"Existing arXiv documents: {existing}")

    processed = 0
    rounds = 0
    while processed + existing < DOC_CAP and rounds < 3:
        for cat in ARXIV_CATEGORIES:
            if processed + existing >= DOC_CAP:
                break
            src = create_arxiv_category_source(cat, max_results=PER_CATEGORY)
            stats = pipeline.ingest_source(src)
            logger.info(f"{cat}: +{stats['documents_processed']} docs, +{stats['chunks_created']} chunks")
            processed += stats['documents_processed']
        rounds += 1

    logger.info(f"Total new arXiv docs ingested: {processed}")

    # Embed and rebuild FAISS
    ep = EmbeddingProcessor()
    emb_stats = ep.process_unembedded_chunks(db)
    logger.info(f"Embeddings: {emb_stats}")

    im = IndexManager(db)
    idx_stats = im.build_index(force_rebuild=True)
    logger.info(f"Index: {idx_stats}")

    # Final corpus stats
    from retrieval import RetrievalEngine
    re = RetrievalEngine(db, im)
    print(re.get_corpus_stats())
