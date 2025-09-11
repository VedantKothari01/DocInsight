#!/usr/bin/env python3
"""
DocInsight Complete Demo Script
===============================

This script sets up everything needed for DocInsight and then launches the Streamlit app.
Run this script to get the full DocInsight experience:

1. Downloads and processes real datasets (PAWS, Wikipedia, arXiv)
2. Builds semantic search index with FAISS
3. Launches production-ready Streamlit interface
4. Ready for document upload and plagiarism detection

Usage:
    python docinsight_demo.py [--target-size SIZE] [--quick-demo]
"""

import sys
import os
import time
import argparse
import subprocess
from pathlib import Path
import logging

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_banner():
    """Print DocInsight banner."""
    banner = """
    ╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║               🔍 DocInsight Enhanced                 ║
    ║          AI-Powered Plagiarism Detection             ║
    ║                                                      ║
    ║   🚀 Real Dataset Integration (PAWS, Wikipedia,      ║
    ║      arXiv) - NO HARDCODED CORPUS                   ║
    ║   🧠 Advanced ML Pipeline with Confidence Scoring   ║
    ║   ⚡ Production-Ready Web Interface                  ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝
    """
    print(banner)


def check_dependencies():
    """Check if all required dependencies are installed."""
    logger.info("Checking dependencies...")
    
    missing_deps = []
    
    try:
        import sentence_transformers
        logger.info("✅ sentence-transformers available")
    except ImportError:
        missing_deps.append("sentence-transformers")
    
    try:
        import faiss
        logger.info("✅ FAISS available")
    except ImportError:
        missing_deps.append("faiss-cpu")
    
    try:
        import datasets
        logger.info("✅ datasets available")
    except ImportError:
        missing_deps.append("datasets")
    
    try:
        import wikipedia
        logger.info("✅ wikipedia available")
    except ImportError:
        missing_deps.append("wikipedia")
    
    try:
        import streamlit
        logger.info("✅ streamlit available")
    except ImportError:
        missing_deps.append("streamlit")
    
    if missing_deps:
        logger.error(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.info("Please install missing dependencies:")
        logger.info(f"pip install {' '.join(missing_deps)}")
        return False
    
    logger.info("✅ All dependencies available!")
    return True


def setup_docinsight(target_size: int = 10000, quick_demo: bool = False):
    """Set up DocInsight with real datasets."""
    logger.info("Setting up DocInsight...")
    
    # Adjust target size for quick demo
    if quick_demo:
        target_size = min(1000, target_size)
        logger.info(f"Quick demo mode: using smaller corpus size ({target_size})")
    
    try:
        # Import our modules
        from corpus_builder import CorpusIndex
        from enhanced_pipeline import PlagiarismDetector
        
        logger.info("📊 Initializing corpus builder...")
        corpus_index = CorpusIndex(target_size=target_size)
        
        logger.info("🌐 Loading real datasets (PAWS, Wikipedia, arXiv)...")
        logger.info("   This may take a few minutes on first run...")
        
        start_time = time.time()
        success = corpus_index.load_or_build()
        build_time = time.time() - start_time
        
        if not success:
            logger.error("❌ Failed to build corpus from real datasets")
            return False
        
        logger.info(f"✅ Built corpus with {len(corpus_index.sentences)} sentences in {build_time:.1f}s")
        
        # Build embeddings and search index
        logger.info("🧠 Building semantic embeddings...")
        start_time = time.time()
        corpus_index.build_embeddings()
        embed_time = time.time() - start_time
        logger.info(f"✅ Built embeddings in {embed_time:.1f}s")
        
        logger.info("⚡ Building FAISS search index...")
        start_time = time.time()
        corpus_index.build_index()
        index_time = time.time() - start_time
        logger.info(f"✅ Built search index in {index_time:.1f}s")
        
        # Test the pipeline
        logger.info("🔍 Testing plagiarism detector...")
        detector = PlagiarismDetector(corpus_index)
        
        # Test with a sample sentence
        test_sentence = "Machine learning algorithms can identify patterns in data."
        result = detector.analyze_sentence(test_sentence)
        logger.info(f"✅ Test analysis complete - Score: {result.fused_score:.3f}, Confidence: {result.confidence}")
        
        # Test document analysis
        test_doc = "Machine learning is powerful. Neural networks can learn complex patterns."
        doc_result = detector.analyze_document(test_doc)
        logger.info(f"✅ Document analysis test complete - {doc_result['overall_stats']['total_sentences']} sentences analyzed")
        
        total_setup_time = build_time + embed_time + index_time
        logger.info(f"🎉 DocInsight setup complete! Total time: {total_setup_time:.1f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return False


def launch_streamlit():
    """Launch the Streamlit application."""
    logger.info("🚀 Launching Streamlit application...")
    
    try:
        # Check if streamlit_app.py exists
        app_file = Path("streamlit_app.py")
        if not app_file.exists():
            logger.error("❌ streamlit_app.py not found in current directory")
            return False
        
        # Launch Streamlit
        logger.info("🌐 Starting web interface...")
        logger.info("📱 The application will open in your default browser")
        logger.info("🔗 URL: http://localhost:8501")
        
        # Run streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"], check=True)
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to launch Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("👋 Streamlit application stopped by user")
        return True


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="DocInsight Complete Demo - Real Dataset Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python docinsight_demo.py                     # Full setup with 10K sentences
  python docinsight_demo.py --quick-demo        # Quick demo with 1K sentences
  python docinsight_demo.py --target-size 50000 # Large corpus with 50K sentences
        """
    )
    
    parser.add_argument(
        "--target-size", 
        type=int, 
        default=10000,
        help="Target corpus size (number of sentences). Default: 10000"
    )
    
    parser.add_argument(
        "--quick-demo", 
        action="store_true",
        help="Quick demo mode with smaller corpus (1000 sentences)"
    )
    
    parser.add_argument(
        "--skip-setup", 
        action="store_true",
        help="Skip setup and directly launch Streamlit (assumes setup already done)"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Please install missing dependencies and try again")
        return 1
    
    # Setup DocInsight unless skipping
    if not args.skip_setup:
        logger.info("🛠️  Setting up DocInsight with real datasets...")
        if not setup_docinsight(args.target_size, args.quick_demo):
            logger.error("❌ Setup failed. Please check the logs and try again.")
            return 1
    else:
        logger.info("⏭️  Skipping setup as requested")
    
    # Launch Streamlit
    logger.info("🎯 Setup complete! Launching web interface...")
    
    print("\n" + "="*60)
    print("🎉 DOCINSIGHT IS READY!")
    print("="*60)
    print("📋 What happens next:")
    print("  1. Streamlit web interface will open in your browser")
    print("  2. Upload any document (TXT, PDF, DOCX)")
    print("  3. Get comprehensive plagiarism analysis report")
    print("  4. No manual corpus upload needed!")
    print("\n💡 Features available:")
    print("  ✅ Real dataset integration (PAWS, Wikipedia, arXiv)")
    print("  ✅ Advanced ML pipeline with confidence scoring")
    print("  ✅ Complete document analysis (not just top sentences)")
    print("  ✅ Semantic similarity + stylometry analysis")
    print("  ✅ Downloadable reports (JSON + summary)")
    print("="*60)
    
    if not launch_streamlit():
        logger.error("❌ Failed to launch Streamlit application")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())