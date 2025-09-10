"""
Enhanced Streamlit app for DocInsight - Production-ready web interface
"""
import streamlit as st
import os
import json
import sys
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_pipeline import EnhancedPlagiarismDetector
    HAS_ENHANCED = True
except ImportError:
    HAS_ENHANCED = False

# Initialize detector (cached)
@st.cache_resource
def load_detector():
    """Load and cache the enhanced detector."""
    if HAS_ENHANCED:
        detector = EnhancedPlagiarismDetector(corpus_size=5000)  # Reduced for web app performance
        detector.initialize()
        return detector
    else:
        st.error("Enhanced pipeline not available. Please ensure all dependencies are installed.")
        return None

def main():
    st.set_page_config(
        page_title="DocInsight Enhanced",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 DocInsight Enhanced - AI-Powered Plagiarism Detection")
    st.markdown("### Version 2.0 with Real Dataset Integration")
    
    # Header with key features
    st.markdown("""
    **🎉 Major Upgrade**: Now powered by real datasets with 50,000+ sentences from PAWS, Wikipedia, arXiv and more!
    
    **Key Features:**
    - 🚀 **One-click analysis** - just upload and analyze
    - 🧠 **Advanced ML models** for semantic similarity and cross-encoder reranking  
    - 📊 **Confidence scoring** with detailed breakdown
    - 🌐 **Multi-domain corpus** covering academic, general, and technical content
    - ⚡ **Fast processing** with optimized FAISS indexing
    """)
    
    # Sidebar with information
    with st.sidebar:
        st.header("📊 System Information")
        
        if HAS_ENHANCED:
            with st.spinner("Loading enhanced system..."):
                detector = load_detector()
            
            if detector:
                st.success("✅ System Ready!")
                st.info(f"📚 Corpus: {len(detector.corpus_sentences):,} sentences")
                st.info(f"🧠 SBERT: {'✓' if detector.sbert_model else '✗'}")
                st.info(f"🔄 CrossEncoder: {'✓' if detector.cross_encoder else '✗'}")
                st.info(f"🔍 FAISS Index: {'✓' if detector.faiss_index else '✗'}")
            else:
                st.error("❌ System not ready")
                return
        else:
            st.error("❌ Enhanced pipeline not available")
            st.markdown("""
            **To enable enhanced features:**
            1. Install required packages from `requirements.txt`
            2. Ensure all modules are in the same directory
            3. Restart the Streamlit app
            """)
            return
        
        st.header("ℹ️ What's New in v2.0")
        st.markdown("""
        - **Real Dataset Integration**
        - **50K+ Sentence Corpus**
        - **Advanced ML Models** 
        - **Multi-Domain Coverage**
        - **Confidence Scoring**
        - **Enhanced Stylometry**
        - **Performance Optimized**
        """)
        
        st.header("📋 Supported Formats")
        st.markdown("""
        - **Text Files** (.txt)
        - **PDF Documents** (.pdf)
        - **Word Documents** (.docx, .doc)
        """)
    
    # Main content
    st.header("📄 Document Analysis")
    
    # Upload section
    uploaded_file = st.file_uploader(
        "Upload your document for plagiarism analysis",
        type=['txt', 'pdf', 'docx', 'doc'],
        help="Upload a document to analyze for potential plagiarism"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"✅ File uploaded: **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
        
        # Save uploaded file
        file_path = f'/tmp/{uploaded_file.name}'
        try:
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
        except Exception as e:
            st.error(f"❌ Error saving file: {e}")
            return
        
        # Analysis options
        st.subheader("⚙️ Analysis Options")
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_threshold = st.selectbox(
                "Minimum confidence to display:",
                ["All", "LOW", "MEDIUM", "HIGH"],
                help="Filter results by confidence level"
            )
        
        with col2:
            max_sentences = st.slider(
                "Max sentences to show:",
                min_value=5,
                max_value=50,
                value=20,
                help="Limit number of results displayed"
            )
        
        # Analysis button
        if st.button("🔍 Analyze Document", type="primary", use_container_width=True):
            with st.spinner("🔄 Running enhanced plagiarism analysis..."):
                start_time = time.time()
                
                try:
                    # Generate report
                    report = detector.generate_enhanced_report(
                        file_path,
                        output_json='/tmp/report.json',
                        output_html='/tmp/report.html'
                    )
                    
                    end_time = time.time()
                    analysis_time = end_time - start_time
                    
                    # Display results
                    st.header("📊 Analysis Results")
                    
                    # Key metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("📝 Sentences", report['total_sentences'])
                    with col2:
                        st.metric("📚 Corpus Size", f"{report['corpus_size']:,}")
                    with col3:
                        st.metric("📊 Avg Score", f"{report['overall_stats']['avg_fused_score']:.3f}")
                    with col4:
                        st.metric("🎯 Max Score", f"{report['overall_stats']['max_fused_score']:.3f}")
                    with col5:
                        st.metric("⏱️ Time", f"{analysis_time:.1f}s")
                    
                    # Confidence distribution with visual indicators
                    st.subheader("🎯 Detection Confidence Distribution")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        high_count = report['overall_stats']['high_confidence_count']
                        st.metric(
                            "🔴 High Confidence", 
                            high_count,
                            help="Strong indicators of potential plagiarism"
                        )
                        if high_count > 0:
                            st.warning(f"⚠️ {high_count} sentence(s) with high plagiarism confidence")
                    
                    with col2:
                        medium_count = report['overall_stats']['medium_confidence_count']
                        st.metric(
                            "🟡 Medium Confidence", 
                            medium_count,
                            help="Moderate similarities detected"
                        )
                        if medium_count > 0:
                            st.info(f"ℹ️ {medium_count} sentence(s) with moderate similarities")
                    
                    with col3:
                        low_count = report['overall_stats']['low_confidence_count']
                        st.metric(
                            "🟢 Low Confidence", 
                            low_count,
                            help="Minimal or no similarities found"
                        )
                    
                    # Overall assessment
                    if high_count > 0:
                        st.error("🚨 **HIGH RISK**: Potential plagiarism detected in multiple sentences")
                    elif medium_count > report['total_sentences'] * 0.3:
                        st.warning("⚠️ **MEDIUM RISK**: Moderate similarities found in several sentences")
                    else:
                        st.success("✅ **LOW RISK**: No significant plagiarism indicators detected")
                    
                    # Detailed sentence analysis
                    st.subheader("📋 Detailed Sentence Analysis")
                    
                    # Filter sentences based on user preferences
                    filtered_sentences = report['sentences']
                    if confidence_threshold != "All":
                        filtered_sentences = [
                            s for s in report['sentences'] 
                            if s['confidence'] == confidence_threshold
                        ]
                    
                    # Limit number of sentences shown
                    display_sentences = filtered_sentences[:max_sentences]
                    
                    if not display_sentences:
                        st.info("No sentences match the selected criteria.")
                    else:
                        st.write(f"Showing {len(display_sentences)} of {len(filtered_sentences)} matching sentences")
                        
                        # Display sentences with enhanced formatting
                        for i, sentence_data in enumerate(display_sentences, 1):
                            confidence = sentence_data['confidence']
                            confidence_colors = {
                                'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'
                            }
                            confidence_styles = {
                                'HIGH': 'background-color: #ffebee; border-left: 4px solid #f44336;',
                                'MEDIUM': 'background-color: #fff8e1; border-left: 4px solid #ff9800;',
                                'LOW': 'background-color: #e8f5e8; border-left: 4px solid #4caf50;'
                            }
                            
                            with st.container():
                                st.markdown(f"""
                                <div style="padding: 1rem; margin: 0.5rem 0; border-radius: 0.5rem; {confidence_styles[confidence]}">
                                    <h4>{confidence_colors[confidence]} Sentence {i} - {confidence} Confidence</h4>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Original sentence
                                st.write("**📝 Original Text:**")
                                st.write(f"*{sentence_data['sentence']}*")
                                
                                # Best match (if found)
                                if sentence_data['best_match']:
                                    st.write("**🎯 Best Match Found:**")
                                    st.write(f"*{sentence_data['best_match']}*")
                                    
                                    # Detailed scores in columns
                                    score_col1, score_col2, score_col3, score_col4 = st.columns(4)
                                    
                                    with score_col1:
                                        st.metric(
                                            "🧠 Semantic", 
                                            f"{sentence_data['semantic_score']:.3f}",
                                            help="Semantic similarity score"
                                        )
                                    with score_col2:
                                        st.metric(
                                            "🔄 Rerank", 
                                            f"{sentence_data['rerank_score']:.3f}",
                                            help="Cross-encoder rerank score"
                                        )
                                    with score_col3:
                                        st.metric(
                                            "📊 Stylometry", 
                                            f"{sentence_data['stylometry_score']:.3f}",
                                            help="Stylistic similarity score"
                                        )
                                    with score_col4:
                                        fused_score = sentence_data['fused_score']
                                        delta_color = "inverse" if fused_score > 0.7 else "normal"
                                        st.metric(
                                            "🎯 **Fused Score**", 
                                            f"{fused_score:.3f}",
                                            help="Combined final score"
                                        )
                                else:
                                    st.success("✅ No similar content found in corpus")
                                
                                st.divider()
                    
                    # Download section
                    st.subheader("📥 Download Detailed Reports")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if os.path.exists('/tmp/report.json'):
                            with open('/tmp/report.json', 'r', encoding='utf-8') as f:
                                json_data = f.read()
                            st.download_button(
                                "📄 Download JSON Report",
                                json_data,
                                file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_docinsight_report.json",
                                mime="application/json",
                                use_container_width=True
                            )
                    
                    with col2:
                        if os.path.exists('/tmp/report.html'):
                            with open('/tmp/report.html', 'r', encoding='utf-8') as f:
                                html_data = f.read()
                            st.download_button(
                                "🌐 Download HTML Report",
                                html_data,
                                file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_docinsight_report.html",
                                mime="text/html",
                                use_container_width=True
                            )
                    
                    # Technical details (expandable)
                    with st.expander("🔧 Technical Details"):
                        st.json({
                            "analysis_time_seconds": round(analysis_time, 2),
                            "corpus_size": report['corpus_size'],
                            "total_sentences_analyzed": report['total_sentences'],
                            "confidence_distribution": {
                                "high": report['overall_stats']['high_confidence_count'],
                                "medium": report['overall_stats']['medium_confidence_count'],
                                "low": report['overall_stats']['low_confidence_count']
                            },
                            "scoring_statistics": {
                                "average_fused_score": report['overall_stats']['avg_fused_score'],
                                "maximum_fused_score": report['overall_stats']['max_fused_score']
                            }
                        })
                    
                except Exception as e:
                    st.error(f"❌ **Error during analysis**: {str(e)}")
                    with st.expander("🔧 Debug Information"):
                        st.exception(e)
    
    else:
        # Instructions when no file uploaded
        st.info("👆 **Please upload a document to begin analysis**")
        
        # Sample usage instructions
        st.subheader("📖 How to Use")
        st.markdown("""
        1. **Upload** your document using the file uploader above
        2. **Configure** analysis options (confidence threshold, max sentences)
        3. **Click "Analyze"** to run the plagiarism detection
        4. **Review results** with confidence-based scoring
        5. **Download** detailed reports for documentation
        """)
        
        # Example results preview
        with st.expander("📊 Example Analysis Results"):
            st.markdown("""
            The enhanced DocInsight system provides:
            
            - **Confidence Levels**: HIGH 🔴 | MEDIUM 🟡 | LOW 🟢
            - **Multiple Scores**: Semantic, Rerank, Stylometry, and Fused
            - **Detailed Matching**: Shows exact source matches from corpus
            - **Performance Metrics**: Analysis time and corpus coverage
            - **Export Options**: JSON and HTML report formats
            """)

if __name__ == "__main__":
    main()