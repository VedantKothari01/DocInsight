"""
Clean Streamlit app for DocInsight - Production-ready web interface
"""
import streamlit as st
import os
import json
import time
from pathlib import Path
from typing import Optional

# Defensive imports
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

from corpus_builder import CorpusIndex
from enhanced_pipeline import PlagiarismDetector

# Initialize detector (cached)
@st.cache_resource
def load_detector():
    """Load and cache the plagiarism detector for production use."""
    try:
        # Auto-detect available cached corpus size
        cache_dir = Path("corpus_cache")
        detected_size = None
        if cache_dir.exists():
            corpus_files = sorted(cache_dir.glob("corpus_*.json"))
            if corpus_files:
                # pick the largest available corpus by size suffix
                try:
                    detected_size = max(int(p.stem.split("_")[1]) for p in corpus_files)
                except Exception:
                    detected_size = None
        # Fallback order if nothing detected
        candidate_sizes = [s for s in [detected_size, 50000, 10000, 5000, 1000, 500] if s]
        if not candidate_sizes:
            candidate_sizes = [10000]

        # Try sizes until one is ready
        corpus_index = None
        for size in candidate_sizes:
            ci = CorpusIndex(target_size=size)
            if ci.is_ready_for_production():
                corpus_index = ci
                break
        if corpus_index is None:
            # Not "ready" (missing flag) – still try to load if all assets exist
            for size in candidate_sizes:
                ci = CorpusIndex(target_size=size)
                if ci._is_fully_cached():
                    corpus_index = ci
                    break
        if corpus_index is None:
            corpus_index = CorpusIndex(target_size=candidate_sizes[0])
        
        # Check if system is ready for production
        if corpus_index.is_ready_for_production() or corpus_index._is_fully_cached():
            st.info("🚀 Loading DocInsight (production-ready)...")
            success = corpus_index.load_for_production()
            
            if success:
                st.success(f"✅ Ready! {len(corpus_index.sentences):,} sentences loaded from real datasets")
                st.info("📊 **Data Sources**: PAWS paraphrase dataset, Wikipedia articles, arXiv abstracts")
                
                # Create detector
                detector = PlagiarismDetector(corpus_index)
                st.session_state['detector'] = detector
                return detector
            else:
                st.error("❌ Failed to load production assets")
                return None
        else:
            # System not set up yet
            st.error("⚠️ **DocInsight Not Set Up Yet**")
            st.markdown("""
            DocInsight requires one-time setup before use. Please run:
            
            ```bash
            python setup_docinsight.py
            ```
            
            This will:
            - Download real datasets (PAWS, Wikipedia, arXiv)
            - Build semantic embeddings  
            - Create search indices
            - Prepare system for instant analysis
            
            After setup, refresh this page to use DocInsight.
            """)
            return None
            
    except Exception as e:
        st.error(f"Failed to initialize DocInsight: {e}")
        st.markdown("""
        **Troubleshooting:**
        1. Run setup first: `python setup_docinsight.py`
        2. Check dependencies: `pip install -r requirements.txt`
        3. Verify network access for dataset downloads
        """)
        return None

def extract_text_from_file(uploaded_file) -> Optional[str]:
    """Extract text from uploaded file."""
    file_type = uploaded_file.name.lower().split('.')[-1]
    
    try:
        if file_type == 'txt':
            return str(uploaded_file.read(), "utf-8")
        
        elif file_type == 'pdf' and HAS_PYMUPDF:
            # Save uploaded file temporarily
            temp_path = Path(f"./cache/{uploaded_file.name}")
            temp_path.parent.mkdir(exist_ok=True)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Extract text using PyMuPDF
            doc = fitz.open(temp_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            # Clean up
            temp_path.unlink()
            return text
        
        elif file_type in ['docx', 'doc'] and HAS_DOCX:
            # Save uploaded file temporarily
            temp_path = Path(f"./cache/{uploaded_file.name}")
            temp_path.parent.mkdir(exist_ok=True)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Extract text using python-docx
            doc = Document(temp_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            # Clean up
            temp_path.unlink()
            return text
        
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
            
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return None

def display_analysis_results(results: dict):
    """Display analysis results in a nice format."""
    overall_stats = results['overall_stats']
    sentence_analyses = results['sentence_analyses']
    
    # Overall statistics
    st.subheader("📊 Overall Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sentences", overall_stats['total_sentences'])
    
    with col2:
        st.metric("Avg Similarity Score", f"{overall_stats['avg_fused_score']:.3f}")
    
    with col3:
        st.metric("Max Similarity Score", f"{overall_stats['max_fused_score']:.3f}")
    
    with col4:
        high_risk_pct = overall_stats['high_risk_ratio'] * 100
        st.metric("High Risk %", f"{high_risk_pct:.1f}%")
    
    # Risk summary
    st.subheader("🚨 Risk Assessment")
    
    high_count = overall_stats['high_confidence_count']
    medium_count = overall_stats['medium_confidence_count']
    low_count = overall_stats['low_confidence_count']
    
    if high_count > 0:
        st.error(f"⚠️ **HIGH RISK**: {high_count} sentences with high similarity detected!")
    elif medium_count > 0:
        st.warning(f"⚠️ **MEDIUM RISK**: {medium_count} sentences with medium similarity detected.")
    else:
        st.success("✅ **LOW RISK**: No high-similarity content detected.")
    
    # Detailed results
    st.subheader("🔍 Detailed Sentence Analysis")
    
    for i, analysis in enumerate(sentence_analyses):
        sentence = analysis['sentence']
        confidence = analysis['confidence']
        fused_score = analysis['fused_score']
        semantic_score = analysis.get('semantic_score', 0.0)
        stylometry_similarity = analysis.get('stylometry_similarity', 0.0)
        cross_encoder_score = analysis.get('cross_encoder_score', 0.0)
        academic_indicators = analysis.get('academic_indicators', {})
        matches = analysis['matches']
        
        # Color code based on confidence
        if confidence == "HIGH":
            st.error(f"**Sentence {i+1}** (Final Score: {fused_score:.3f}) - {confidence} RISK")
        elif confidence == "MEDIUM":
            st.warning(f"**Sentence {i+1}** (Final Score: {fused_score:.3f}) - {confidence} RISK")
        else:
            st.info(f"**Sentence {i+1}** (Final Score: {fused_score:.3f}) - {confidence} RISK")
        
        st.write(f"*Text:* {sentence}")
        
        # Display component scores (Semantic + Stylometric Analysis)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🧠 Semantic Score", f"{semantic_score:.3f}")
        with col2:
            st.metric("✍️ Stylometric Score", f"{stylometry_similarity:.3f}")
        with col3:
            st.metric("🎯 Cross-Encoder Score", f"{cross_encoder_score:.3f}")
        
        # Display stylometric analysis (Academic Writing Features)
        if academic_indicators:
            with st.expander("📊 Stylometric Analysis (Academic Writing Features)"):
                st.markdown("**Academic Writing Style Analysis:**")
                
                col1, col2 = st.columns(2)
                with col1:
                    if 'academic_word_ratio' in academic_indicators:
                        st.metric("📚 Academic Vocabulary", f"{academic_indicators['academic_word_ratio']:.3f}")
                    if 'citation_density' in academic_indicators:
                        st.metric("📖 Citation Density", f"{academic_indicators['citation_density']:.3f}")
                    if 'flesch_kincaid_grade' in academic_indicators:
                        st.metric("📝 Reading Level", f"{academic_indicators['flesch_kincaid_grade']:.1f}")
                
                with col2:
                    if 'passive_voice_ratio' in academic_indicators:
                        st.metric("🔄 Passive Voice", f"{academic_indicators['passive_voice_ratio']:.3f}")
                    if 'perplexity_estimate' in academic_indicators:
                        st.metric("🤖 AI Perplexity", f"{academic_indicators['perplexity_estimate']:.1f}")
                    if 'repetition_score' in academic_indicators:
                        st.metric("🔁 Repetition Score", f"{academic_indicators['repetition_score']:.3f}")
        
        # Similar matches
        if matches:
            with st.expander(f"🔍 View {len(matches)} similar matches found"):
                for j, match in enumerate(matches):
                    confidence_emoji = "🔴" if match['confidence'] == "HIGH" else "🟡" if match['confidence'] == "MEDIUM" else "🟢"
                    st.write(f"  **{j+1}.** {confidence_emoji} **Similarity: {match['similarity']:.3f}** ({match['confidence']} confidence)")
                    st.write(f"      *{match['text']}*")
        else:
            st.write("   ℹ️ No significant matches found")
        
        st.write("---")

def main():
    st.set_page_config(
        page_title="DocInsight Enhanced",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 DocInsight Enhanced - AI-Powered Plagiarism Detection")
    st.markdown("### 🚀 Production-Ready with Real Dataset Integration")
    
    # Header with key features
    st.markdown("""
    **🎉 Real Dataset Integration**: Now powered by **50,000+ sentences** from PAWS, Wikipedia, and arXiv!
    
    **🌟 Key Features:**
    - 🚀 **One-click analysis** - just upload your document
    - 🧠 **Semantic Analysis** - Advanced ML models with SBERT embeddings + cross-encoder reranking
    - ✍️ **Stylometric Analysis** - 15+ academic writing features (vocabulary, citations, passive voice, AI detection)
    - 📊 **Dual-Engine Detection** - Combines semantic similarity + writing style analysis (beats Turnitin's approach!)
    - 🎯 **Component Scoring** - See individual semantic, stylometric, and cross-encoder scores
    - 📝 **Academic Indicators** - Reading level, citation density, academic vocabulary usage
    - 🤖 **AI Detection Features** - Perplexity estimation, repetition analysis, coherence scoring
    - 🌐 **Real datasets** - PAWS paraphrases, Wikipedia articles, academic papers
    - ⚡ **FAISS indexing** for sub-second similarity search
    - 📝 **Comprehensive reports** with downloadable JSON and summary formats
    - 🔒 **No hardcoded corpus** - purely data-driven detection
    """)
    
    st.info("💡 **RESEARCH ADVANTAGE**: DocInsight uses both semantic similarity AND stylometric analysis - the two areas where Turnitin falls short!")
    
    # Load detector
    detector = load_detector()
    
    if detector is None:
        st.error("❌ Failed to initialize DocInsight. Please check the logs and try again.")
        return
    
    # File upload section
    st.subheader("📄 Upload Your Document")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['txt', 'pdf', 'docx', 'doc'],
        help="Supported formats: TXT, PDF, DOCX, DOC"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Extract text
        with st.spinner("Extracting text from document..."):
            text = extract_text_from_file(uploaded_file)
        
        if text:
            st.success(f"✅ Extracted {len(text)} characters")
            
            # Show preview
            with st.expander("📖 Document Preview"):
                st.text_area("Content preview", text[:1000] + "..." if len(text) > 1000 else text, height=200)
            
            # Analysis button
            if st.button("🔍 Analyze for Plagiarism", type="primary"):
                with st.spinner("Analyzing document... This may take a moment."):
                    try:
                        results = detector.analyze_document(text)
                        
                        if 'error' in results:
                            st.error(f"Analysis failed: {results['error']}")
                        else:
                            st.success("✅ Analysis complete!")
                            display_analysis_results(results)
                            
                            # Export options
                            st.subheader("💾 Export Results")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # JSON export
                                json_str = json.dumps(results, indent=2, ensure_ascii=False)
                                st.download_button(
                                    label="📄 Download JSON Report",
                                    data=json_str,
                                    file_name=f"plagiarism_report_{uploaded_file.name}.json",
                                    mime="application/json"
                                )
                            
                            with col2:
                                # Summary export
                                summary = f"""DocInsight Plagiarism Analysis Report
================================================

Document: {uploaded_file.name}
Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

OVERALL STATISTICS:
- Total Sentences Analyzed: {results['overall_stats']['total_sentences']}
- Average Similarity Score: {results['overall_stats']['avg_fused_score']:.3f}
- Maximum Similarity Score: {results['overall_stats']['max_fused_score']:.3f}
- High Risk Sentences: {results['overall_stats']['high_confidence_count']}
- Medium Risk Sentences: {results['overall_stats']['medium_confidence_count']}
- Low Risk Sentences: {results['overall_stats']['low_confidence_count']}
- High Risk Percentage: {results['overall_stats']['high_risk_ratio']*100:.1f}%

RISK ASSESSMENT:
{
    "HIGH RISK: Immediate attention required!" if results['overall_stats']['high_confidence_count'] > 0 
    else "MEDIUM RISK: Review recommended." if results['overall_stats']['medium_confidence_count'] > 0 
    else "LOW RISK: No significant issues detected."
}

Generated by DocInsight Enhanced v2.0
"""
                                st.download_button(
                                    label="📋 Download Summary Report",
                                    data=summary,
                                    file_name=f"plagiarism_summary_{uploaded_file.name}.txt",
                                    mime="text/plain"
                                )
                    
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {e}")
        else:
            st.error("❌ Failed to extract text from the uploaded file.")
    
    # Information section
    st.sidebar.markdown("""
    ## ℹ️ About DocInsight Enhanced
    
    **Version 2.0** with **REAL dataset integration** - no hardcoded corpus!
    
    ### 🔄 What's New:
    - **50,000+ Real Sentences**: From PAWS, Wikipedia, and arXiv datasets
    - **Complete Document Analysis**: Every sentence analyzed and scored
    - **Advanced ML Pipeline**: SentenceTransformers + Cross-encoder + Stylometry
    - **Zero Manual Input**: No corpus upload needed - fully automated
    - **Confidence Scoring**: HIGH/MEDIUM/LOW risk levels with detailed explanations
    
    ### 📊 Data Sources:
    - **PAWS Dataset**: Paraphrase detection training data
    - **Wikipedia**: 25+ topic areas for general knowledge
    - **arXiv Papers**: Academic abstracts from CS, AI, ML, Physics, Math
    
    ### 🛡️ Privacy Notice:
    - Documents processed locally only
    - No data stored permanently  
    - Analysis results not shared externally
    
    ### 🎯 Best Results:
    - Upload complete documents (not fragments)
    - English text works best
    - Academic and technical content optimal
    - Supports TXT, PDF, DOCX formats
    """)
    
    # Add corpus statistics
    detector = st.session_state.get('detector')
    if detector and hasattr(detector, 'corpus_index'):
        st.sidebar.markdown("### 📈 Current Corpus Stats")
        st.sidebar.metric("Total Sentences", len(detector.corpus_index.sentences))
        sample = detector.corpus_index.get_random_sample(3)
        if sample:
            st.sidebar.markdown("**Sample corpus content:**")
            for i, sent in enumerate(sample, 1):
                st.sidebar.markdown(f"{i}. *{sent[:80]}...*")

if __name__ == "__main__":
    main()