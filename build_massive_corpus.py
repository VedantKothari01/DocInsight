#!/usr/bin/env python3
"""
Massive Academic Corpus Builder

Builds a comprehensive academic corpus (4000-5000 documents) from specific domains.
This is a one-time build that creates a pre-built corpus for the project.

Usage:
    python build_massive_corpus.py
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any
import json

from db import DatabaseManager
from ingestion import IngestionPipeline, create_wiki_search_source, create_arxiv_category_source
from index import IndexManager
from embeddings import EmbeddingProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('corpus_build.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Academic domains configuration
ACADEMIC_DOMAINS = {
    'computer_science': {
        'wiki_topics': [
            'Artificial intelligence', 'Machine learning', 'Computer vision', 'Natural language processing',
            'Deep learning', 'Neural networks', 'Data structures', 'Algorithms', 'Software engineering',
            'Computer networks', 'Cybersecurity', 'Cryptography', 'Database systems', 'Operating systems',
            'Programming languages', 'Human-computer interaction', 'Information theory', 'Computational complexity',
            'Distributed systems', 'Computer graphics', 'Robotics', 'Computer architecture'
        ],
        'arxiv_categories': [
            'cs.AI', 'cs.CL', 'cs.CV', 'cs.LG', 'cs.NE', 'cs.DS', 'cs.DC', 'cs.CR', 'cs.DB', 'cs.OS',
            'cs.PL', 'cs.HC', 'cs.IT', 'cs.CC', 'cs.SE', 'cs.NI', 'cs.GR', 'cs.RO', 'cs.AR'
        ]
    },
    'mathematics': {
        'wiki_topics': [
            'Linear algebra', 'Calculus', 'Statistics', 'Probability theory', 'Number theory',
            'Topology', 'Differential equations', 'Mathematical analysis', 'Abstract algebra',
            'Geometry', 'Combinatorics', 'Graph theory', 'Optimization theory', 'Functional analysis',
            'Real analysis', 'Complex analysis', 'Set theory', 'Logic', 'Discrete mathematics'
        ],
        'arxiv_categories': [
            'math.AG', 'math.AT', 'math.CA', 'math.CO', 'math.CT', 'math.CV', 'math.DG', 'math.DS',
            'math.FA', 'math.GM', 'math.GN', 'math.GT', 'math.HO', 'math.IT', 'math.KT', 'math.LO',
            'math.MG', 'math.NT', 'math.OA', 'math.OC', 'math.PR', 'math.QA', 'math.RT', 'math.SG',
            'math.SP', 'math.ST'
        ]
    },
    'physics': {
        'wiki_topics': [
            'Quantum mechanics', 'Relativity', 'Thermodynamics', 'Electromagnetism', 'Particle physics',
            'Astrophysics', 'Condensed matter physics', 'Nuclear physics', 'Optics', 'Acoustics',
            'Fluid dynamics', 'Statistical mechanics', 'Quantum field theory', 'String theory',
            'Cosmology', 'Plasma physics', 'Solid state physics', 'Atomic physics', 'Molecular physics'
        ],
        'arxiv_categories': [
            'physics.acc-ph', 'physics.app-ph', 'physics.ao-ph', 'physics.atom-ph', 'physics.bio-ph',
            'physics.chem-ph', 'physics.class-ph', 'physics.comp-ph', 'physics.data-an', 'physics.flu-dyn',
            'physics.gen-ph', 'physics.geo-ph', 'physics.hist-ph', 'physics.ins-det', 'physics.med-ph',
            'physics.optics', 'physics.plasm-ph', 'physics.pop-ph', 'physics.soc-ph', 'physics.space-ph'
        ]
    },
    'biology': {
        'wiki_topics': [
            'Molecular biology', 'Cell biology', 'Genetics', 'Evolution', 'Ecology', 'Biochemistry',
            'Microbiology', 'Immunology', 'Neuroscience', 'Developmental biology', 'Bioinformatics',
            'Systems biology', 'Synthetic biology', 'Biotechnology', 'Pharmacology', 'Toxicology',
            'Physiology', 'Anatomy', 'Pathology', 'Epidemiology'
        ],
        'arxiv_categories': [
            'q-bio.BM', 'q-bio.CB', 'q-bio.GN', 'q-bio.MN', 'q-bio.NC', 'q-bio.OT', 'q-bio.PE', 'q-bio.QM', 'q-bio.SC', 'q-bio.TO'
        ]
    },
    'chemistry': {
        'wiki_topics': [
            'Organic chemistry', 'Inorganic chemistry', 'Physical chemistry', 'Analytical chemistry',
            'Biochemistry', 'Theoretical chemistry', 'Computational chemistry', 'Materials chemistry',
            'Environmental chemistry', 'Medicinal chemistry', 'Polymer chemistry', 'Electrochemistry',
            'Thermochemistry', 'Quantum chemistry', 'Surface chemistry', 'Nuclear chemistry'
        ],
        'arxiv_categories': [
            'physics.chem-ph', 'cond-mat.mtrl-sci'
        ]
    },
    'engineering': {
        'wiki_topics': [
            'Mechanical engineering', 'Electrical engineering', 'Civil engineering', 'Chemical engineering',
            'Aerospace engineering', 'Biomedical engineering', 'Environmental engineering', 'Materials engineering',
            'Computer engineering', 'Industrial engineering', 'Nuclear engineering', 'Petroleum engineering',
            'Systems engineering', 'Control engineering', 'Telecommunications engineering'
        ],
        'arxiv_categories': [
            'eess.AS', 'eess.IV', 'eess.SP', 'eess.SY', 'physics.app-ph'
        ]
    },
    'economics': {
        'wiki_topics': [
            'Microeconomics', 'Macroeconomics', 'Econometrics', 'Behavioral economics', 'Development economics',
            'International economics', 'Labor economics', 'Public economics', 'Financial economics',
            'Industrial organization', 'Game theory', 'Economic history', 'Economic geography',
            'Environmental economics', 'Health economics', 'Urban economics', 'Agricultural economics'
        ],
        'arxiv_categories': [
            'econ.EM', 'econ.GN', 'econ.TH', 'q-fin.CP', 'q-fin.EC', 'q-fin.GN', 'q-fin.MF', 'q-fin.PM', 'q-fin.PR', 'q-fin.RM', 'q-fin.ST', 'q-fin.TR'
        ]
    },
    'psychology': {
        'wiki_topics': [
            'Cognitive psychology', 'Social psychology', 'Developmental psychology', 'Clinical psychology',
            'Behavioral psychology', 'Neuropsychology', 'Personality psychology', 'Educational psychology',
            'Industrial psychology', 'Forensic psychology', 'Health psychology', 'Environmental psychology',
            'Experimental psychology', 'Psychometrics', 'Abnormal psychology', 'Positive psychology'
        ],
        'arxiv_categories': [
            'q-bio.NC', 'physics.soc-ph'
        ]
    }
}

MAX_TOTAL_DOCUMENTS = 500


def build_massive_corpus():
    """Build a massive academic corpus from multiple domains"""
    
    logger.info("Starting massive academic corpus build...")
    start_time = time.time()
    
    # Initialize database and pipeline
    db_manager = DatabaseManager()
    pipeline = IngestionPipeline(db_manager)
    
    total_stats = {
        'domains_processed': 0,
        'documents_processed': 0,
        'chunks_created': 0,
        'errors': [],
        'domain_stats': {}
    }
    
    # Process each academic domain
    for domain_name, domain_config in ACADEMIC_DOMAINS.items():
        logger.info(f"Processing domain: {domain_name}")
        domain_start = time.time()
        
        domain_stats = {
            'wiki_documents': 0,
            'arxiv_documents': 0,
            'wiki_chunks': 0,
            'arxiv_chunks': 0,
            'errors': []
        }
        
        try:
            # Process Wikipedia topics
            logger.info(f"Processing {len(domain_config['wiki_topics'])} Wikipedia topics...")
            for topic in domain_config['wiki_topics']:
                try:
                    if total_stats['documents_processed'] >= MAX_TOTAL_DOCUMENTS:
                        break
                    # Request fewer results to favor depth across domains
                    source = create_wiki_search_source(topic, max_results=8)
                    stats = pipeline.ingest_source(source)
                    domain_stats['wiki_documents'] += stats['documents_processed']
                    domain_stats['wiki_chunks'] += stats['chunks_created']
                    logger.info(f"Processed Wikipedia topic '{topic}': {stats['documents_processed']} docs, {stats['chunks_created']} chunks")
                except Exception as e:
                    error_msg = f"Failed to process Wikipedia topic '{topic}': {e}"
                    domain_stats['errors'].append(error_msg)
                    logger.error(error_msg)
                    continue
                if total_stats['documents_processed'] + domain_stats['wiki_documents'] + domain_stats['arxiv_documents'] >= MAX_TOTAL_DOCUMENTS:
                    break
            
            # Process arXiv categories
            logger.info(f"Processing {len(domain_config['arxiv_categories'])} arXiv categories...")
            for category in domain_config['arxiv_categories']:
                try:
                    if total_stats['documents_processed'] >= MAX_TOTAL_DOCUMENTS:
                        break
                    # 10 per category to spread coverage
                    source = create_arxiv_category_source(category, max_results=10)
                    stats = pipeline.ingest_source(source)
                    domain_stats['arxiv_documents'] += stats['documents_processed']
                    domain_stats['arxiv_chunks'] += stats['chunks_created']
                    logger.info(f"Processed arXiv category '{category}': {stats['documents_processed']} docs, {stats['chunks_created']} chunks")
                except Exception as e:
                    error_msg = f"Failed to process arXiv category '{category}': {e}"
                    domain_stats['errors'].append(error_msg)
                    logger.error(error_msg)
                    continue
                if total_stats['documents_processed'] + domain_stats['wiki_documents'] + domain_stats['arxiv_documents'] >= MAX_TOTAL_DOCUMENTS:
                    break
            
            domain_time = time.time() - domain_start
            logger.info(f"Completed domain '{domain_name}' in {domain_time:.1f}s: "
                       f"{domain_stats['wiki_documents'] + domain_stats['arxiv_documents']} docs, "
                       f"{domain_stats['wiki_chunks'] + domain_stats['arxiv_chunks']} chunks")
            
            total_stats['domain_stats'][domain_name] = domain_stats
            total_stats['domains_processed'] += 1
            total_stats['documents_processed'] += domain_stats['wiki_documents'] + domain_stats['arxiv_documents']
            total_stats['chunks_created'] += domain_stats['wiki_chunks'] + domain_stats['arxiv_chunks']
            
        except Exception as e:
            error_msg = f"Failed to process domain '{domain_name}': {e}"
            total_stats['errors'].append(error_msg)
            logger.error(error_msg)
            continue

        if total_stats['documents_processed'] >= MAX_TOTAL_DOCUMENTS:
            logger.info(f"Reached MAX_TOTAL_DOCUMENTS={MAX_TOTAL_DOCUMENTS}. Stopping domain processing.")
            break
    
    # Generate embeddings for all chunks
    logger.info("Generating embeddings for all chunks...")
    embedding_processor = EmbeddingProcessor()
    embedding_stats = embedding_processor.process_unembedded_chunks(db_manager)
    logger.info(f"Generated embeddings: {embedding_stats['chunks_processed']} chunks processed")
    
    # Build FAISS index
    logger.info("Building FAISS index...")
    index_manager = IndexManager(db_manager)
    index_stats = index_manager.build_index(force_rebuild=True)
    logger.info(f"Index built: {index_stats['chunks_indexed']} chunks indexed")
    
    # Save build statistics
    build_time = time.time() - start_time
    total_stats['build_time_seconds'] = build_time
    total_stats['embedding_stats'] = embedding_stats
    total_stats['index_stats'] = index_stats
    
    # Save statistics to file
    with open('corpus_build_stats.json', 'w') as f:
        json.dump(total_stats, f, indent=2)
    
    logger.info(f"Massive corpus build completed in {build_time:.1f}s")
    logger.info(f"Total documents: {total_stats['documents_processed']}")
    logger.info(f"Total chunks: {total_stats['chunks_created']}")
    logger.info(f"Domains processed: {total_stats['domains_processed']}")
    
    return total_stats

if __name__ == "__main__":
    stats = build_massive_corpus()
    print(f"\n🎉 Corpus build completed!")
    print(f"📊 Documents: {stats['documents_processed']}")
    print(f"📄 Chunks: {stats['chunks_created']}")
    print(f"🏛️ Domains: {stats['domains_processed']}")
    print(f"⏱️ Time: {stats['build_time_seconds']:.1f}s")
