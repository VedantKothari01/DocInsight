#!/usr/bin/env python3
import os
import json
import pickle
from pathlib import Path
import logging

# Set env to minimize NumExpr behavior before any other imports
os.environ.setdefault('NUMEXPR_MAX_THREADS', '1')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise SystemExit(f"Failed to import sentence_transformers: {e}")

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False


def main(target_size: int = 20000, cache_dir: str = 'corpus_cache') -> int:
    cache = Path(cache_dir)
    corpus_path = cache / f"corpus_{target_size}.json"
    embeddings_path = cache / f"embeddings_{target_size}.pkl"
    index_path = cache / f"faiss_index_{target_size}.bin"

    if not corpus_path.exists():
        logger.error(f"Corpus not found: {corpus_path}")
        return 1

    logger.info(f"Loading corpus: {corpus_path}")
    with corpus_path.open('r', encoding='utf-8') as f:
        sentences = json.load(f)

    if not isinstance(sentences, list) or not sentences:
        logger.error("Corpus file is not a non-empty list")
        return 1

    logger.info(f"Encoding {len(sentences):,} sentences with all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode in batches to control memory
    batch_size = 256
    all_embeddings = []
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start:start+batch_size]
        emb = model.encode(batch, convert_to_numpy=True)
        all_embeddings.append(emb)
        if (start // batch_size) % 50 == 0:
            logger.info(f"Encoded {min(start+batch_size, len(sentences)):,}/{len(sentences):,}")

    import numpy as np
    embeddings = np.vstack(all_embeddings)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Normalize for cosine similarity
    if HAS_FAISS:
        faiss.normalize_L2(embeddings)

    logger.info(f"Saving embeddings to {embeddings_path}")
    with embeddings_path.open('wb') as f:
        pickle.dump(embeddings, f)

    if HAS_FAISS:
        dim = embeddings.shape[1]
        logger.info(f"Building FAISS IndexFlatIP of dim {dim}")
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype('float32'))
        logger.info(f"Index size: {index.ntotal}")
        logger.info(f"Saving index to {index_path}")
        faiss.write_index(index, str(index_path))
    else:
        logger.warning("FAISS not available; index will not be created.")

    logger.info("Done building 20k assets.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
