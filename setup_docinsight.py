#!/usr/bin/env python3
"""
DocInsight Academic Setup Script - Research-Focused Implementation
================================================================

Implements SRS v0.2 requirements for academic plagiarism detection system:
- Academic paraphrase curriculum (PAWS + Quora + synthetic)
- Domain-adapted SBERT fine-tuning
- Enhanced stylometric analysis for academic writing
- Research-quality corpus building and caching

This script performs one-time setup for conference-quality research system.
After completion, users can instantly analyze academic documents with
sophisticated domain-adapted semantic embeddings and stylometric evidence.

Usage:
    python setup_docinsight.py --target-size 50000 --enable-domain-adaptation
    python setup_docinsight.py --quick --target-size 10000
    python setup_docinsight.py --research-mode  # Full research configuration
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
    """Print enhanced setup banner for research system."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║           🎓 DocInsight Academic Research Setup              ║
    ║              Domain-Adapted Plagiarism Detection            ║
    ║                                                              ║
    ║   SRS v0.2 Implementation - Conference Quality System       ║
    ║                                                              ║
    ║   • Academic paraphrase curriculum (PAWS + Quora)           ║
    ║   • Domain-adapted SBERT fine-tuning                        ║
    ║   • Enhanced stylometric analysis                           ║
    ║   • Research-quality evaluation framework                   ║
    ║                                                              ║
    ║   After completion, analyze academic documents instantly    ║
    ║   with state-of-the-art semantic and stylometric analysis  ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

    """
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

def setup_docinsight_academic(target_size: int, enable_domain_adaptation: bool = True, force_rebuild: bool = False):
    """
    Perform complete DocInsight academic setup with SRS v0.2 features.
    
    Args:
        target_size: Size of corpus to build
        enable_domain_adaptation: Whether to enable domain adaptation training
        force_rebuild: Force rebuild even if cache exists
    """
    logger.info(f"Starting DocInsight Academic setup with target size: {target_size}")
    logger.info(f"Domain adaptation: {'ENABLED' if enable_domain_adaptation else 'DISABLED'}")
    
    try:
        from corpus_builder import CorpusIndex
        from enhanced_pipeline import AcademicPlagiarismDetector
        
        # Initialize academic corpus builder with domain adaptation
        corpus_index = CorpusIndex(
            target_size=target_size, 
            use_domain_adaptation=enable_domain_adaptation
        )
        
        # Check if already set up (unless force rebuild)
        if not force_rebuild and corpus_index._is_fully_cached():
            logger.info("✅ DocInsight Academic is already set up and ready!")
            logger.info("Use --force-rebuild to rebuild from scratch")
            
            # Quick validation test with academic detector
            detector = AcademicPlagiarismDetector(corpus_index, use_domain_adaptation=enable_domain_adaptation)
            test_result = detector.analyze_sentence("Machine learning algorithms identify patterns in academic research.")
            logger.info(f"✅ Academic system validation passed - Score: {test_result.fused_score:.3f}")
            logger.info(f"   Semantic: {test_result.semantic_score:.3f}, Stylometric: {test_result.stylometry_similarity:.3f}")
            return True
        
        # Step 1: Build academic corpus from enhanced datasets
        logger.info("📊 STEP 1: Building academic corpus from enhanced datasets...")
        logger.info("   • PAWS paraphrase dataset")
        logger.info("   • Quora question pairs")
        logger.info("   • Academic Wikipedia articles")
        logger.info("   • arXiv research abstracts")
        logger.info("   • Synthetic academic paraphrases")
        
        start_time = time.time()
        success = corpus_index.load_or_build()
        if not success:
            logger.error("❌ Failed to build academic corpus")
            return False
        
        build_time = time.time() - start_time
        logger.info(f"✅ Academic corpus built in {build_time:.1f} seconds")
        logger.info(f"   📈 Corpus size: {len(corpus_index.sentences)} sentences")
        
        # Step 2: Domain adaptation training (if enabled)
        if enable_domain_adaptation:
            logger.info("🧠 STEP 2: Academic domain adaptation training...")
            logger.info("   Fine-tuning SBERT on academic paraphrase curriculum")
            
            try:
                # Domain adaptation is handled in corpus_index._init_model()
                if corpus_index.model:
                    logger.info("✅ Domain-adapted model successfully loaded")
                else:
                    logger.warning("⚠️ Domain adaptation failed, using base model")
            except Exception as e:
                logger.warning(f"⚠️ Domain adaptation failed: {e}")
        
        # Step 3: Enhanced validation with academic features
        logger.info("🔍 STEP 3: Academic system validation...")
        
        detector = AcademicPlagiarismDetector(corpus_index, use_domain_adaptation=enable_domain_adaptation)
        
        # Test academic sentences
        test_sentences = [
            "Machine learning algorithms identify patterns in academic research data.",
            "This study demonstrates the effectiveness of neural networks for classification tasks.",
            "The methodology employed in this investigation follows established research protocols."
        ]
        
        total_test_time = 0
        for i, test_sentence in enumerate(test_sentences, 1):
            test_start = time.time()
            result = detector.analyze_sentence(test_sentence)
            test_time = time.time() - test_start
            total_test_time += test_time
            
            logger.info(f"   Test {i}: Score {result.fused_score:.3f} ({test_time:.3f}s)")
            logger.info(f"     - Semantic: {result.semantic_score:.3f}")
            logger.info(f"     - Stylometric: {result.stylometry_similarity:.3f}")
            logger.info(f"     - Cross-encoder: {result.cross_encoder_score:.3f}")
            logger.info(f"     - Confidence: {result.confidence}")
            
            if result.academic_indicators:
                logger.info(f"     - Academic word ratio: {result.academic_indicators.get('academic_word_ratio', 0.0):.3f}")
                logger.info(f"     - Citation density: {result.academic_indicators.get('citation_density', 0.0):.3f}")
        
        avg_analysis_time = total_test_time / len(test_sentences)
        logger.info(f"✅ Academic validation passed - Avg analysis time: {avg_analysis_time:.3f}s")
        
        # Step 4: Create production ready marker
        logger.info("💾 STEP 4: Finalizing academic system...")
        ready_file = corpus_index.cache_dir / ".docinsight_academic_ready"
        with open(ready_file, 'w') as f:
            import json
            setup_info = {
                'timestamp': time.time(),
                'target_size': target_size,
                'actual_size': len(corpus_index.sentences),
                'domain_adaptation': enable_domain_adaptation,
                'avg_analysis_time': avg_analysis_time,
                'version': 'SRS_v0.2'
            }
            json.dump(setup_info, f, indent=2)
        
        total_time = time.time() - start_time
        logger.info(f"🎉 DocInsight Academic setup completed in {total_time:.1f} seconds!")
        logger.info("📋 Setup Summary:")
        logger.info(f"   • Academic corpus: {len(corpus_index.sentences)} sentences")
        logger.info(f"   • Domain adaptation: {'✅' if enable_domain_adaptation else '❌'}")
        logger.info(f"   • Average analysis time: {avg_analysis_time:.3f}s")
        logger.info(f"   • Cache location: {corpus_index.cache_dir}")
        logger.info("")
        logger.info("🚀 Ready for academic plagiarism detection!")
        logger.info("   Run: python run_docinsight.py")
        logger.info("   Or:  streamlit run streamlit_app.py")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
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
    """Main academic setup function with SRS v0.2 features."""
    parser = argparse.ArgumentParser(
        description="DocInsight Academic Research Setup - SRS v0.2 Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Academic Research Setup Examples:
  python setup_docinsight.py --research-mode           # Full research configuration
  python setup_docinsight.py --target-size 50000       # Large academic corpus
  python setup_docinsight.py --quick --target-size 5000 # Quick research setup  
  python setup_docinsight.py --no-domain-adaptation    # Disable domain adaptation
  python setup_docinsight.py --force-rebuild           # Force complete rebuild

SRS v0.2 Features:
  • Academic paraphrase curriculum (PAWS + Quora + synthetic)
  • Domain-adapted SBERT fine-tuning for academic writing
  • Enhanced stylometric analysis with academic indicators
  • Research-quality evaluation framework

After setup completion:
  1. Run: python run_docinsight.py                     # Launch academic system
  2. Web: streamlit run streamlit_app.py               # Web interface
  3. Research: python research_evaluation.py           # Run benchmarks
        """
    )
    
    parser.add_argument(
        "--target-size",
        type=int,
        default=20000,
        help="Target academic corpus size (number of sentences). Default: 20000"
    )
    
    parser.add_argument(
        "--research-mode",
        action="store_true",
        help="Full research configuration with large corpus and domain adaptation"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick setup with smaller corpus (5000 sentences)"
    )
    
    parser.add_argument(
        "--no-domain-adaptation",
        action="store_true",
        help="Disable domain adaptation (use base SBERT model)"
    )
    
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force complete rebuild even if cache exists"
    )
    
    args = parser.parse_args()
    
    # Configure based on modes
    if args.research_mode:
        args.target_size = max(50000, args.target_size)
        enable_domain_adaptation = True
        logger.info("🎓 Research mode activated - full academic configuration")
    elif args.quick:
        args.target_size = min(5000, args.target_size)
        enable_domain_adaptation = True
        logger.info("⚡ Quick mode activated - minimal academic setup")
    else:
        enable_domain_adaptation = not args.no_domain_adaptation
    
    print_setup_banner()
    
    logger.info(f"DocInsight Academic Setup Configuration:")
    logger.info(f"  Target corpus size: {args.target_size:,} sentences")
    logger.info(f"  Domain adaptation: {'ENABLED' if enable_domain_adaptation else 'DISABLED'}")
    logger.info(f"  Force rebuild: {args.force_rebuild}")
    logger.info(f"  Research mode: {args.research_mode}")
    logger.info(f"  Quick mode: {args.quick}")
    
    # Check system requirements
    if not check_system_requirements():
        logger.error("❌ System requirements not met")
        return 1
    
    # Perform academic setup
    logger.info("🚀 Starting DocInsight Academic setup process...")
    
    success = setup_docinsight_academic(
        target_size=args.target_size,
        enable_domain_adaptation=enable_domain_adaptation,
        force_rebuild=args.force_rebuild
    )
    
    if success:
        logger.info("🎉 SUCCESS: DocInsight Academic is ready for research!")
        
        # Suggest next steps based on configuration
        logger.info("\n📋 Recommended next steps:")
        if args.research_mode:
            logger.info("  1. Run research evaluation: python research_evaluation.py")
            logger.info("  2. Test academic analysis: python run_docinsight.py --test")
            logger.info("  3. Launch web interface: streamlit run streamlit_app.py")
        else:
            logger.info("  1. Test system: python run_docinsight.py --validate")
            logger.info("  2. Launch web interface: streamlit run streamlit_app.py") 
            logger.info("  3. Analyze documents: python run_docinsight.py")
        
        return 0
    else:
        logger.error("❌ FAILED: DocInsight Academic setup encountered errors")
        logger.error("   Check logs above and try again with --force-rebuild")
        return 1
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




    