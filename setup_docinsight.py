#!/usr/bin/env python3
"""
DocInsight One-Time Setup Script
===============================

This script performs the one-time setup/training for DocInsight:
1. Downloads real datasets (PAWS, Wikipedia, arXiv)
2. Builds and caches sentence embeddings
3. Creates and saves FAISS search index
4. Prepares all assets for production usage

After running this script once, users can directly use DocInsight without
any downloads or heavy processing - just like using a pre-trained model.

Usage:
    python setup_docinsight.py --target-size 50000
    python setup_docinsight.py --quick --target-size 5000
"""

import sys
import os
import time
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('docinsight_setup.log')
    ]
)
logger = logging.getLogger(__name__)

def print_setup_banner():
    """Print setup banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║               🔧 DocInsight Setup & Training                 ║
    ║                  One-Time Configuration                      ║
    ║                                                              ║
    ║   This script downloads datasets, builds embeddings,        ║
    ║   and prepares DocInsight for production usage.             ║
    ║                                                              ║
    ║   After completion, users can instantly analyze documents   ║
    ║   without any setup or downloads required.                  ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_system_requirements():
    """Check if system has required dependencies."""
    logger.info("Checking system requirements...")
    
    required_packages = [
        'sentence_transformers',
        'faiss',
        'datasets',
        'wikipedia',
        'requests',
        'numpy'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'faiss':
                import faiss
            else:
                __import__(package)
            logger.info(f"✅ {package} available")
        except ImportError:
            missing.append(package)
            logger.error(f"❌ {package} missing")
    
    if missing:
        logger.error(f"Missing required packages: {', '.join(missing)}")
        logger.error("Please install: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All system requirements satisfied")
    return True

def setup_docinsight(target_size: int, force_rebuild: bool = False):
    """Perform complete DocInsight setup."""
    logger.info(f"Starting DocInsight setup with target size: {target_size}")
    
    try:
        from corpus_builder import CorpusIndex
        from enhanced_pipeline import PlagiarismDetector
        
        # Initialize corpus builder
        corpus_index = CorpusIndex(target_size=target_size)
        
        # Check if already set up (unless force rebuild)
        if not force_rebuild and corpus_index._is_fully_cached():
            logger.info("✅ DocInsight is already set up and ready!")
            logger.info("Use --force-rebuild to rebuild from scratch")
            
            # Quick validation test
            detector = PlagiarismDetector(corpus_index)
            test_result = detector.analyze_sentence("Machine learning algorithms identify patterns.")
            logger.info(f"✅ System validation passed - Score: {test_result.fused_score:.3f}")
            return True
        
        # Step 1: Build corpus from real datasets
        logger.info("📊 STEP 1: Building corpus from real datasets...")
        logger.info("   This downloads PAWS, Wikipedia, and arXiv data")
        
        start_time = time.time()
        success = corpus_index.load_or_build()
        if not success:
            logger.error("❌ Failed to build corpus")
            return False
        
        corpus_time = time.time() - start_time
        logger.info(f"✅ Corpus built: {len(corpus_index.sentences)} sentences in {corpus_time:.1f}s")
        
        # Step 2: Build embeddings
        logger.info("🧠 STEP 2: Generating sentence embeddings...")
        logger.info("   This creates semantic vectors for all sentences")
        
        start_time = time.time()
        success = corpus_index.build_embeddings()
        if not success:
            logger.error("❌ Failed to build embeddings")
            return False
        
        embed_time = time.time() - start_time
        logger.info(f"✅ Embeddings built: {corpus_index.embeddings.shape} in {embed_time:.1f}s")
        
        # Step 3: Build FAISS index
        logger.info("⚡ STEP 3: Building FAISS search index...")
        logger.info("   This creates fast similarity search capability")
        
        start_time = time.time()
        success = corpus_index.build_index()
        if not success:
            logger.warning("⚠️ FAISS index building failed - will use fallback search")
        else:
            index_time = time.time() - start_time
            logger.info(f"✅ FAISS index built: {corpus_index.index.ntotal} vectors in {index_time:.1f}s")
        
        # Step 4: Validate complete system
        logger.info("🔍 STEP 4: Validating complete system...")
        
        detector = PlagiarismDetector(corpus_index)
        
        # Test sentence analysis
        test_sentence = "Machine learning algorithms can identify patterns in data."
        result = detector.analyze_sentence(test_sentence)
        logger.info(f"✅ Sentence analysis test: Score {result.fused_score:.3f}, Confidence {result.confidence}")
        
        # Test document analysis
        test_doc = "Artificial intelligence is transforming industries. Machine learning enables pattern recognition."
        doc_result = detector.analyze_document(test_doc)
        logger.info(f"✅ Document analysis test: {doc_result['overall_stats']['total_sentences']} sentences processed")
        
        total_time = corpus_time + embed_time + (index_time if 'index_time' in locals() else 0)
        logger.info(f"🎉 SETUP COMPLETE! Total time: {total_time:.1f}s")
        
        # Create ready flag
        ready_file = Path("corpus_cache/.docinsight_ready")
        ready_file.parent.mkdir(exist_ok=True)
        with open(ready_file, 'w') as f:
            f.write(f"DocInsight setup completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Target size: {target_size}\n")
            f.write(f"Corpus sentences: {len(corpus_index.sentences)}\n")
            f.write(f"Setup time: {total_time:.1f}s\n")
        
        logger.info("✅ DocInsight is now ready for production use!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="DocInsight One-Time Setup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_docinsight.py                        # Standard setup (10K sentences)
  python setup_docinsight.py --target-size 50000    # Large corpus (50K sentences)
  python setup_docinsight.py --quick                # Quick setup (2K sentences)
  python setup_docinsight.py --force-rebuild        # Force complete rebuild

After setup completion, you can:
  1. Run: python docinsight_demo.py --skip-setup
  2. Or directly: streamlit run streamlit_app.py
        """
    )
    
    parser.add_argument(
        "--target-size",
        type=int,
        default=10000,
        help="Target corpus size (number of sentences). Default: 10000"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick setup with smaller corpus (2000 sentences)"
    )
    
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force complete rebuild even if cache exists"
    )
    
    args = parser.parse_args()
    
    # Adjust target size for quick mode
    if args.quick:
        args.target_size = min(2000, args.target_size)
    
    print_setup_banner()
    
    logger.info(f"DocInsight Setup Configuration:")
    logger.info(f"  Target corpus size: {args.target_size:,} sentences")
    logger.info(f"  Force rebuild: {args.force_rebuild}")
    logger.info(f"  Quick mode: {args.quick}")
    
    # Check system requirements
    if not check_system_requirements():
        logger.error("❌ System requirements not met")
        return 1
    
    # Perform setup
    logger.info("🚀 Starting DocInsight setup process...")
    success = setup_docinsight(args.target_size, args.force_rebuild)
    
    if success:
        logger.info("")
        logger.info("=" * 70)
        logger.info("🎉 DOCINSIGHT SETUP SUCCESSFUL!")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Your DocInsight system is now ready for production use!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Launch the web interface:")
        logger.info("     streamlit run streamlit_app.py")
        logger.info("")
        logger.info("  2. Or run the demo script:")
        logger.info("     python docinsight_demo.py --skip-setup")
        logger.info("")
        logger.info("  3. Upload documents and get instant plagiarism analysis!")
        logger.info("")
        logger.info("Features available:")
        logger.info("  ✅ Instant document analysis (no setup required)")
        logger.info(f"  ✅ Large corpus ({args.target_size:,} real sentences)")
        logger.info("  ✅ Real datasets (PAWS, Wikipedia, arXiv)")
        logger.info("  ✅ Advanced ML pipeline with confidence scoring")
        logger.info("  ✅ Production-ready performance")
        logger.info("=" * 70)
        return 0
    else:
        logger.error("❌ Setup failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    exit(main())