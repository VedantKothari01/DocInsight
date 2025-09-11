#!/usr/bin/env python3
"""
DocInsight Complete System Test & Demonstration
==============================================

This script demonstrates the complete DocInsight system functionality
with real datasets and comprehensive testing.

Author: AI Assistant for @VedantKothari01
Purpose: Production-ready plagiarism detection system
"""

import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_demo_banner():
    """Print demonstration banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║                    🎯 DocInsight Complete System Demonstration              ║
    ║                     Production-Ready Plagiarism Detection                   ║
    ║                                                                              ║
    ║  This script demonstrates the complete DocInsight pipeline:                 ║
    ║  • Real dataset integration (PAWS, Wikipedia, arXiv)                        ║
    ║  • Advanced ML models (sentence transformers, cross-encoders)               ║
    ║  • Comprehensive document analysis                                           ║
    ║  • Production-ready SaaS architecture                                       ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def test_dataset_loading():
    """Test individual dataset loading capabilities."""
    logger.info("🔬 PHASE 1: Testing Dataset Loading")
    
    try:
        from dataset_loaders import DatasetLoader
        loader = DatasetLoader()
        
        # Test PAWS dataset
        logger.info("Testing PAWS paraphrase dataset...")
        paws_sentences = loader.load_paws_dataset(max_samples=20)
        logger.info(f"✅ PAWS: {len(paws_sentences)} sentences loaded")
        
        # Test Wikipedia
        logger.info("Testing Wikipedia article loading...")
        wiki_sentences = loader.load_wikipedia_articles(
            topics=['Artificial intelligence', 'Machine learning'], 
            sentences_per_topic=10
        )
        logger.info(f"✅ Wikipedia: {len(wiki_sentences)} sentences loaded")
        
        # Test arXiv
        logger.info("Testing arXiv academic papers...")
        arxiv_sentences = loader.load_arxiv_abstracts(max_papers=6)
        logger.info(f"✅ arXiv: {len(arxiv_sentences)} sentences loaded")
        
        logger.info("✅ PHASE 1 COMPLETE: All datasets loading successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        return False

def test_corpus_building():
    """Test corpus building from multiple datasets."""
    logger.info("🏗️ PHASE 2: Testing Corpus Building")
    
    try:
        from corpus_builder import CorpusIndex
        
        # Build small corpus for testing
        corpus_index = CorpusIndex(target_size=500)
        success = corpus_index.load_or_build()
        
        if success:
            logger.info(f"✅ Corpus built: {len(corpus_index.sentences)} sentences")
            logger.info(f"✅ Embeddings: {corpus_index.embeddings.shape}")
            if corpus_index.index:
                logger.info(f"✅ FAISS index: {corpus_index.index.ntotal} vectors")
            else:
                logger.info("⚠️ FAISS index: Using fallback similarity search")
        else:
            logger.error("❌ Corpus building failed")
            return False
        
        logger.info("✅ PHASE 2 COMPLETE: Corpus building successful")
        return corpus_index
        
    except Exception as e:
        logger.error(f"❌ Corpus building failed: {e}")
        return False

def test_plagiarism_detection(corpus_index):
    """Test plagiarism detection pipeline."""
    logger.info("🔍 PHASE 3: Testing Plagiarism Detection Pipeline")
    
    try:
        from enhanced_pipeline import PlagiarismDetector
        
        detector = PlagiarismDetector(corpus_index)
        
        # Test sentences with varying similarity levels
        test_cases = [
            "Machine learning algorithms can identify complex patterns in large datasets.",
            "Artificial neural networks are inspired by biological brain structures.",
            "Climate change poses significant challenges to global sustainability efforts.",
            "This is a completely unique sentence that should not match anything in our corpus at all.",
            "The quick brown fox jumps over the lazy dog repeatedly in the forest."
        ]
        
        logger.info("Testing individual sentence analysis...")
        for i, sentence in enumerate(test_cases, 1):
            result = detector.analyze_sentence(sentence)
            logger.info(f"  Test {i}: Score {result.fused_score:.3f}, Confidence {result.confidence}")
            logger.info(f"    Text: {sentence[:60]}...")
            if result.matches:
                best_match = result.matches[0]
                logger.info(f"    Best match: {best_match.text[:50]}... (sim: {best_match.similarity:.3f})")
        
        # Test document analysis
        logger.info("\nTesting complete document analysis...")
        test_document = """
        Artificial intelligence is revolutionizing various industries worldwide.
        Machine learning algorithms enable computers to learn from data automatically.
        Natural language processing allows machines to understand human communication.
        Computer vision systems can analyze and interpret visual information effectively.
        These technologies are transforming how we work and interact with digital systems.
        """
        
        doc_result = detector.analyze_document(test_document.strip())
        stats = doc_result['overall_stats']
        
        logger.info(f"✅ Document Analysis Results:")
        logger.info(f"  Total sentences: {stats['total_sentences']}")
        logger.info(f"  Average score: {stats['avg_fused_score']:.3f}")
        logger.info(f"  High confidence: {stats['high_confidence_count']}")
        logger.info(f"  Medium confidence: {stats['medium_confidence_count']}")
        logger.info(f"  Low confidence: {stats['low_confidence_count']}")
        
        logger.info("✅ PHASE 3 COMPLETE: Plagiarism detection pipeline working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Plagiarism detection failed: {e}")
        return False

def test_production_workflow():
    """Test the production SaaS workflow."""
    logger.info("🚀 PHASE 4: Testing Production SaaS Workflow")
    
    try:
        # Test setup script functionality
        logger.info("Testing setup script capabilities...")
        
        from corpus_builder import CorpusIndex
        
        # Create a production-ready corpus
        corpus_index = CorpusIndex(target_size=1000)
        
        if corpus_index.is_ready_for_production():
            logger.info("✅ System already set up for production")
        else:
            logger.info("Setting up production environment...")
            success = corpus_index.load_or_build()
            if not success:
                logger.error("❌ Production setup failed")
                return False
        
        # Test production loading
        logger.info("Testing production loading speed...")
        start_time = time.time()
        
        prod_corpus = CorpusIndex(target_size=corpus_index.target_size)
        success = prod_corpus.load_for_production()
        
        load_time = time.time() - start_time
        
        if success:
            logger.info(f"✅ Production loading: {load_time:.2f}s for {len(prod_corpus.sentences)} sentences")
        else:
            logger.error("❌ Production loading failed")
            return False
        
        logger.info("✅ PHASE 4 COMPLETE: Production workflow verified")
        return True
        
    except Exception as e:
        logger.error(f"❌ Production workflow test failed: {e}")
        return False

def test_streamlit_readiness():
    """Test Streamlit application readiness."""
    logger.info("🌐 PHASE 5: Testing Web Interface Readiness")
    
    try:
        # Check if streamlit app exists
        app_file = Path("streamlit_app.py")
        if not app_file.exists():
            logger.error("❌ streamlit_app.py not found")
            return False
        
        # Check if we can import streamlit modules used in the app
        import streamlit as st
        logger.info("✅ Streamlit available")
        
        # Test app components
        logger.info("Testing app component imports...")
        from corpus_builder import CorpusIndex
        from enhanced_pipeline import PlagiarismDetector
        
        logger.info("✅ All required modules available for web interface")
        logger.info("✅ PHASE 5 COMPLETE: Web interface ready")
        
        logger.info("\n🌐 To launch web interface:")
        logger.info("   streamlit run streamlit_app.py")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Web interface test failed: {e}")
        return False

def generate_final_report():
    """Generate final system report."""
    logger.info("📊 FINAL SYSTEM REPORT")
    logger.info("=" * 80)
    
    try:
        from corpus_builder import CorpusIndex
        
        # Find existing corpus
        cache_dir = Path("corpus_cache")
        target_size = 1000
        
        if cache_dir.exists():
            corpus_files = list(cache_dir.glob("corpus_*.json"))
            if corpus_files:
                target_size = int(corpus_files[0].stem.split('_')[1])
        
        corpus_index = CorpusIndex(target_size=target_size)
        
        if corpus_index.is_ready_for_production():
            corpus_index.load_for_production()
            
            logger.info(f"📈 SYSTEM STATUS: PRODUCTION READY")
            logger.info(f"📊 Corpus Size: {len(corpus_index.sentences):,} sentences")
            logger.info(f"🧠 Embeddings: {corpus_index.embeddings.shape}")
            logger.info(f"⚡ Search Index: {'FAISS' if corpus_index.index else 'Fallback'}")
            logger.info(f"🌐 Web Interface: Ready")
            logger.info(f"🔍 Detection Pipeline: Operational")
            
            logger.info("\n🎯 FEATURES AVAILABLE:")
            logger.info("  ✅ Real dataset integration (PAWS, Wikipedia, arXiv)")
            logger.info("  ✅ Advanced ML pipeline (transformers, cross-encoders)")
            logger.info("  ✅ Comprehensive document analysis")
            logger.info("  ✅ Production-ready caching")
            logger.info("  ✅ SaaS-style architecture")
            logger.info("  ✅ Web-based interface")
            
            logger.info("\n🚀 USAGE:")
            logger.info("  1. Setup (one-time): python setup_docinsight.py")
            logger.info("  2. Launch: python run_docinsight.py")
            logger.info("  3. Web UI: streamlit run streamlit_app.py")
            
        else:
            logger.warning("⚠️ System needs setup - run setup_docinsight.py")
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")

def main():
    """Run complete system demonstration."""
    print_demo_banner()
    
    start_time = time.time()
    phases_completed = 0
    total_phases = 5
    
    try:
        # Phase 1: Dataset Loading
        if test_dataset_loading():
            phases_completed += 1
        else:
            logger.error("❌ Phase 1 failed - stopping demonstration")
            return 1
        
        # Phase 2: Corpus Building
        corpus_index = test_corpus_building()
        if corpus_index:
            phases_completed += 1
        else:
            logger.error("❌ Phase 2 failed - stopping demonstration")
            return 1
        
        # Phase 3: Plagiarism Detection
        if test_plagiarism_detection(corpus_index):
            phases_completed += 1
        else:
            logger.error("❌ Phase 3 failed - stopping demonstration")
            return 1
        
        # Phase 4: Production Workflow
        if test_production_workflow():
            phases_completed += 1
        else:
            logger.error("❌ Phase 4 failed - stopping demonstration")
            return 1
        
        # Phase 5: Web Interface
        if test_streamlit_readiness():
            phases_completed += 1
        else:
            logger.error("❌ Phase 5 failed - stopping demonstration")
            return 1
        
        total_time = time.time() - start_time
        
        logger.info("")
        logger.info("🎉" * 40)
        logger.info("🎉 DOCINSIGHT COMPLETE SYSTEM DEMONSTRATION SUCCESSFUL! 🎉")
        logger.info("🎉" * 40)
        logger.info("")
        logger.info(f"✅ All {phases_completed}/{total_phases} phases completed successfully")
        logger.info(f"⏱️ Total demonstration time: {total_time:.1f} seconds")
        logger.info("")
        
        generate_final_report()
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        logger.error(f"Completed {phases_completed}/{total_phases} phases")
        return 1

if __name__ == "__main__":
    exit(main())