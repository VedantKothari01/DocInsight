#!/usr/bin/env python3
"""
DocInsight Production Runner
===========================

Fast launcher for DocInsight after setup is complete.
This script assumes DocInsight has already been set up using setup_docinsight.py.

Usage:
    python run_docinsight.py                # Launch web interface
    python run_docinsight.py --test         # Run quick functionality test
    python run_docinsight.py --validate     # Validate system integrity
"""

import sys
import os
import argparse
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_production_banner():
    """Print production banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║               🚀 DocInsight Production Runner                ║
    ║                  Ready for Document Analysis                 ║
    ║                                                              ║
    ║   Fast startup • No setup required • Production ready       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_system_ready():
    """Check if DocInsight is ready for production use."""
    try:
        from corpus_builder import CorpusIndex
        
        # Try to find existing corpus files to determine target size
        cache_dir = Path("corpus_cache")
        if cache_dir.exists():
            corpus_files = list(cache_dir.glob("corpus_*.json"))
            if corpus_files:
                # Extract target size from first corpus file
                corpus_file = corpus_files[0]
                target_size = int(corpus_file.stem.split('_')[1])
                logger.info(f"Found existing corpus with target size: {target_size}")
                
                corpus_index = CorpusIndex(target_size=target_size)
                if corpus_index.is_ready_for_production():
                    logger.info("✅ DocInsight is ready for production use")
                    return True
        
        # Fallback: try common sizes
        for size in [200, 1000, 5000, 10000, 50000]:
            corpus_index = CorpusIndex(target_size=size)
            if corpus_index.is_ready_for_production():
                logger.info(f"✅ DocInsight is ready for production use (size: {size})")
                return True
        
        logger.error("❌ DocInsight is not set up for production")
        logger.error("Please run: python setup_docinsight.py")
        return False
            
    except Exception as e:
        logger.error(f"❌ System check failed: {e}")
        return False

def validate_system():
    """Validate complete system functionality."""
    logger.info("🔍 Validating DocInsight system...")
    
    try:
        from corpus_builder import CorpusIndex
        from enhanced_pipeline import PlagiarismDetector
        
        # Find the correct target size
        cache_dir = Path("corpus_cache")
        target_size = 10000  # default
        
        if cache_dir.exists():
            corpus_files = list(cache_dir.glob("corpus_*.json"))
            if corpus_files:
                target_size = int(corpus_files[0].stem.split('_')[1])
        
        # Load corpus
        corpus_index = CorpusIndex(target_size=target_size)
        if not corpus_index.load_for_production():
            logger.error("❌ Failed to load corpus for production")
            return False
        
        # Test detector
        detector = PlagiarismDetector(corpus_index)
        
        # Test sentence analysis
        test_sentence = "Machine learning algorithms can identify patterns in data efficiently."
        result = detector.analyze_sentence(test_sentence)
        
        if result and result.fused_score is not None:
            logger.info(f"✅ Sentence analysis: Score {result.fused_score:.3f}, Confidence {result.confidence}")
        else:
            logger.error("❌ Sentence analysis failed")
            return False
        
        # Test document analysis
        test_document = """
        Artificial intelligence is transforming various industries.
        Machine learning algorithms enable computers to learn from data.
        Neural networks can model complex relationships in information.
        """
        
        doc_result = detector.analyze_document(test_document.strip())
        
        if doc_result and 'overall_stats' in doc_result:
            total_sentences = doc_result['overall_stats']['total_sentences']
            logger.info(f"✅ Document analysis: {total_sentences} sentences processed")
        else:
            logger.error("❌ Document analysis failed")
            return False
        
        # Check corpus quality
        corpus_size = len(corpus_index.sentences)
        if corpus_size < 1000:
            logger.warning(f"⚠️ Corpus size is small: {corpus_size} sentences")
        else:
            logger.info(f"✅ Corpus quality: {corpus_size:,} sentences available")
        
        # Check embeddings
        if corpus_index.embeddings is not None:
            logger.info(f"✅ Embeddings ready: {corpus_index.embeddings.shape}")
        else:
            logger.warning("⚠️ No embeddings loaded")
        
        # Check FAISS index
        if corpus_index.index is not None:
            logger.info(f"✅ FAISS index ready: {corpus_index.index.ntotal:,} vectors")
        else:
            logger.warning("⚠️ No FAISS index (will use fallback search)")
        
        logger.info("🎉 System validation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_quick_test():
    """Run a quick functionality test."""
    logger.info("🧪 Running quick functionality test...")
    
    try:
        from corpus_builder import CorpusIndex
        from enhanced_pipeline import PlagiarismDetector
        
        # Find the correct target size
        cache_dir = Path("corpus_cache")
        target_size = 10000  # default
        
        if cache_dir.exists():
            corpus_files = list(cache_dir.glob("corpus_*.json"))
            if corpus_files:
                target_size = int(corpus_files[0].stem.split('_')[1])
        
        # Quick load test
        corpus_index = CorpusIndex(target_size=target_size)
        success = corpus_index.load_for_production()
        
        if not success:
            logger.error("❌ Failed to load system")
            return False
        
        # Quick analysis test
        detector = PlagiarismDetector(corpus_index)
        result = detector.analyze_sentence("Neural networks are powerful machine learning models.")
        
        logger.info(f"✅ Quick test passed - Score: {result.fused_score:.3f}")
        logger.info(f"   Corpus: {len(corpus_index.sentences):,} sentences")
        logger.info(f"   Confidence: {result.confidence}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Quick test failed: {e}")
        return False

def launch_streamlit():
    """Launch Streamlit application."""
    logger.info("🌐 Launching DocInsight web interface...")
    
    try:
        # Check if streamlit_app.py exists
        app_file = Path("streamlit_app.py")
        if not app_file.exists():
            logger.error("❌ streamlit_app.py not found")
            return False
        
        logger.info("🚀 Starting web interface at http://localhost:8501")
        logger.info("📱 The application will open in your default browser")
        
        # Launch Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"], check=True)
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to launch Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("👋 Application stopped by user")
        return True

def main():
    """Main production runner."""
    parser = argparse.ArgumentParser(
        description="DocInsight Production Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_docinsight.py            # Launch web interface
  python run_docinsight.py --test     # Quick functionality test
  python run_docinsight.py --validate # Full system validation

Note: This assumes DocInsight setup is already complete.
If not set up, run: python setup_docinsight.py
        """
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick functionality test and exit"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true", 
        help="Validate complete system and exit"
    )
    
    args = parser.parse_args()
    
    print_production_banner()
    
    # Check if system is ready
    if not check_system_ready():
        logger.error("System not ready for production use")
        logger.info("Run setup first: python setup_docinsight.py")
        return 1
    
    # Handle test mode
    if args.test:
        if run_quick_test():
            logger.info("✅ Quick test completed successfully")
            return 0
        else:
            logger.error("❌ Quick test failed")
            return 1
    
    # Handle validation mode
    if args.validate:
        if validate_system():
            logger.info("✅ System validation completed successfully")
            return 0
        else:
            logger.error("❌ System validation failed")
            return 1
    
    # Launch web interface
    logger.info("🎯 DocInsight is ready! Launching web interface...")
    
    print("\n" + "="*60)
    print("🎉 DOCINSIGHT PRODUCTION READY!")
    print("="*60)
    print("📋 What happens next:")
    print("  1. Web interface opens at http://localhost:8501")
    print("  2. Upload any document (TXT, PDF, DOCX)")
    print("  3. Get instant comprehensive plagiarism analysis")
    print("  4. Download detailed reports")
    print("\n💡 Ready features:")
    print("  ✅ Instant startup (no setup/downloads)")
    print("  ✅ Real dataset corpus (PAWS, Wikipedia, arXiv)")
    print("  ✅ Advanced ML pipeline with confidence scoring")
    print("  ✅ Complete document analysis")
    print("  ✅ Production-grade performance")
    print("="*60)
    
    if not launch_streamlit():
        logger.error("❌ Failed to launch web interface")
        return 1
    
    return 0

if __name__ == "__main__":
    import multiprocessing #Prevention of segmentation error
    multiprocessing.set_start_method("forkserver", force=True)
    exit(main())
