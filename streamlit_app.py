"""
DocInsight Streamlit Application

Updated UI to show high-level originality metrics, risk spans, and capped sentence details.
Features document-level aggregation scoring and improved user experience.
"""

import streamlit as st
import json
import os
import tempfile
import logging
from pathlib import Path
import traceback

# Local imports
from enhanced_pipeline import DocumentAnalysisPipeline
from config import MAX_SENTENCE_DISPLAY, TOP_RISK_SPANS_PREVIEW, SUPPORTED_EXTENSIONS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="DocInsight - Document Originality Analysis",
    page_icon="📄",
    layout="wide"
)


def display_originality_metrics(metrics: dict):
    """Display document-level originality metrics"""
    st.header("📊 Originality Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        originality_score = metrics.get('originality_score', 0.0)
        st.metric(
            label="Originality Score", 
            value=f"{originality_score:.1%}",
            delta=None
        )
    
    with col2:
        coverage = metrics.get('plagiarized_coverage', 0.0)
        st.metric(
            label="Plagiarized Coverage",
            value=f"{coverage:.1%}",
            delta=None
        )
    
    with col3:
        severity = metrics.get('severity_index', 0.0)
        st.metric(
            label="Severity Index",
            value=f"{severity:.3f}",
            delta=None
        )
    
    with col4:
        total_sentences = metrics.get('total_sentences', 0)
        st.metric(
            label="Total Sentences",
            value=total_sentences,
            delta=None
        )


def display_sentence_distribution(distribution: dict):
    """Display sentence risk distribution"""
    st.subheader("📈 Risk Distribution")
    
    if distribution:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            high_count = distribution.get('HIGH', 0)
            st.metric(
                label="🔴 High Risk",
                value=high_count,
                delta=None
            )
        
        with col2:
            medium_count = distribution.get('MEDIUM', 0)
            st.metric(
                label="🟡 Medium Risk", 
                value=medium_count,
                delta=None
            )
        
        with col3:
            low_count = distribution.get('LOW', 0)
            st.metric(
                label="🟢 Low Risk",
                value=low_count,
                delta=None
            )


def display_top_risk_spans(top_spans: list):
    """Display top risk spans with expandable details"""
    if not top_spans:
        st.info("No significant risk spans detected.")
        return
    
    st.subheader(f"⚠️ Top {len(top_spans)} Risk Spans")
    
    for i, span in enumerate(top_spans, 1):
        risk_icon = "🔴" if span['risk_level'] == 'HIGH' else "🟡"
        
        with st.expander(f"{risk_icon} Risk Span {i} - {span['risk_level']} (Score: {span['avg_score']:.3f})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Preview:**")
                st.write(span.get('preview_text', 'No preview available'))
            
            with col2:
                st.write("**Details:**")
                st.write(f"- Sentences: {len(span['sentences'])}")
                st.write(f"- Token count: {span['token_count']}")
                st.write(f"- Position: {span['start_index']}-{span['end_index']}")
            
            # Show sentences in this span
            if st.checkbox(f"Show sentences in span {i}", key=f"span_{i}"):
                st.write("**Sentences in this span:**")
                for j, sent_result in enumerate(span['sentences']):
                    sentence = sent_result.get('sentence', '')
                    confidence = sent_result.get('confidence_score', 0.0)
                    best_match = sent_result.get('best_match', '')
                    
                    st.write(f"**{j+1}.** {sentence}")
                    if best_match:
                        st.write(f"   *Similar to:* {best_match} (confidence: {confidence:.3f})")
                    st.write("---")


def display_sentence_details(sentence_results: list, max_display: int = MAX_SENTENCE_DISPLAY):
    """Display detailed sentence analysis (capped to prevent UI overload)"""
    st.subheader("📝 Sentence Analysis Details")
    
    total_sentences = len(sentence_results)
    
    if total_sentences > max_display:
        st.warning(f"Showing first {max_display} of {total_sentences} sentences to prevent UI overload.")
        display_sentences = sentence_results[:max_display]
    else:
        display_sentences = sentence_results
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        risk_filter = st.selectbox(
            "Filter by risk level:",
            options=["All", "HIGH", "MEDIUM", "LOW"],
            index=0
        )
    
    with col2:
        show_details = st.checkbox("Show detailed scores", value=False)
    
    # Apply filter
    if risk_filter != "All":
        display_sentences = [s for s in display_sentences if s.get('risk_level') == risk_filter]
    
    if not display_sentences:
        st.info(f"No sentences found with {risk_filter} risk level.")
        return
    
    # Display sentences
    for i, result in enumerate(display_sentences, 1):
        sentence = result.get('sentence', '')
        risk_level = result.get('risk_level', 'LOW')
        confidence = result.get('confidence_score', 0.0)
        best_match = result.get('best_match', '')
        
        # Choose icon based on risk level
        if risk_level == 'HIGH':
            risk_icon = "🔴"
            risk_color = "#ffebee"
        elif risk_level == 'MEDIUM':
            risk_icon = "🟡"
            risk_color = "#fff8e1"
        else:
            risk_icon = "🟢"
            risk_color = "#e8f5e8"
        
        # Display sentence in colored container
        with st.container():
            st.markdown(
                f"""
                <div style="background-color: {risk_color}; padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>{risk_icon} Sentence {i} ({risk_level})</strong><br>
                    {sentence}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            if best_match:
                st.write(f"**Most similar:** {best_match}")
                st.write(f"**Confidence:** {confidence:.3f}")
            
            if show_details:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"Semantic: {result.get('semantic_score', 0.0):.3f}")
                with col2:
                    st.write(f"Cross-encoder: {result.get('rerank_score', 0.0):.3f}")
                with col3:
                    st.write(f"Stylometry: {result.get('stylometry_score', 0.0):.3f}")
            
            st.write("---")


def create_download_buttons(report_files: dict, analysis_result: dict):
    """Create download buttons for reports"""
    st.subheader("📥 Download Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'html' in report_files and os.path.exists(report_files['html']):
            with open(report_files['html'], 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            st.download_button(
                label="📄 Download HTML Report",
                data=html_content,
                file_name="docinsight_report.html",
                mime="text/html"
            )
    
    with col2:
        if 'json' in report_files and os.path.exists(report_files['json']):
            with open(report_files['json'], 'r', encoding='utf-8') as f:
                json_content = f.read()
            
            st.download_button(
                label="📊 Download JSON Report",
                data=json_content,
                file_name="docinsight_report.json",
                mime="application/json"
            )


def display_processing_info(processing_info: dict):
    """Display information about available processing components"""
    with st.expander("🔧 Processing Information"):
        st.write("**Available Components:**")
        
        components = [
            ("Semantic Search Engine", processing_info.get('semantic_engine_available', False)),
            ("Cross-Encoder Reranker", processing_info.get('cross_encoder_available', False)),
            ("Stylometry Analyzer", processing_info.get('stylometry_available', False))
        ]
        
        for name, available in components:
            status = "✅" if available else "❌"
            st.write(f"{status} {name}")


def main():
    """Main Streamlit application"""
    st.title("📄 DocInsight - Document Originality Analysis")
    st.markdown("Upload a document to analyze its originality and detect potential plagiarism.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a document to analyze",
        type=[ext.lstrip('.') for ext in SUPPORTED_EXTENSIONS],
        help=f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
    )
    
    if uploaded_file is not None:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_path = tmp_file.name
        
        try:
            # Show processing message
            with st.spinner("Analyzing document... This may take a few minutes."):
                # Initialize pipeline and analyze document
                pipeline = DocumentAnalysisPipeline()
                analysis_result = pipeline.analyze_document(temp_path)
                
                # Generate report files
                report_files = pipeline.generate_report_files(analysis_result)
            
            # Display results
            st.success("✅ Analysis completed!")
            
            # Extract key components
            originality_analysis = analysis_result.get('originality_analysis', {})
            originality_metrics = originality_analysis.get('originality_metrics', {})
            top_risk_spans = originality_analysis.get('top_risk_spans', [])
            sentence_results = analysis_result.get('sentence_results', [])
            sentence_distribution = originality_metrics.get('sentence_distribution', {})
            processing_info = analysis_result.get('processing_info', {})
            
            # Display main metrics
            display_originality_metrics(originality_metrics)
            
            # Display distribution and risk spans side by side
            col1, col2 = st.columns([1, 2])
            with col1:
                display_sentence_distribution(sentence_distribution)
            with col2:
                display_top_risk_spans(top_risk_spans)
            
            # Display detailed sentence analysis
            display_sentence_details(sentence_results)
            
            # Download buttons
            create_download_buttons(report_files, analysis_result)
            
            # Processing info
            display_processing_info(processing_info)
            
        except Exception as e:
            st.error(f"❌ Error analyzing document: {str(e)}")
            logger.error(f"Error in document analysis: {e}")
            logger.error(traceback.format_exc())
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary file {temp_path}: {e}")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About DocInsight")
        st.markdown("""
        DocInsight analyzes documents for originality using:
        
        **🔍 Multi-layer Analysis:**
        - Semantic similarity search
        - Cross-encoder reranking  
        - Stylometric feature comparison
        
        **📊 Document-level Metrics:**
        - Originality score (0-100%)
        - Plagiarized content coverage
        - Risk span clustering
        - Severity assessment
        
        **🎯 Risk Classification:**
        - 🔴 **High Risk**: Likely plagiarized content
        - 🟡 **Medium Risk**: Potentially similar content
        - 🟢 **Low Risk**: Original content
        """)
        
        st.header("🚀 Features")
        st.markdown("""
        - Support for PDF, DOCX, and TXT files
        - Real-time similarity detection
        - Detailed sentence-level analysis
        - Downloadable HTML/JSON reports
        - Risk span clustering
        - Processing component status
        """)


if __name__ == "__main__":
    main()