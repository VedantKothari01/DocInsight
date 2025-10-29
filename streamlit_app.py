"""
DocInsight Streamlit Application - Multi-File Upload Support

Supports uploading and analyzing up to 15 documents simultaneously.
"""

import streamlit as st
import json
import os
import tempfile
import logging
from pathlib import Path
import traceback
from typing import List, Dict

# Local imports
from enhanced_pipeline import DocumentAnalysisPipeline
from config import MAX_SENTENCE_DISPLAY, TOP_RISK_SPANS_PREVIEW, SUPPORTED_EXTENSIONS, EXTENDED_CORPUS_ENABLED
from db import get_db_manager
from ingestion import IngestionPipeline, create_wiki_search_source, create_arxiv_source, create_arxiv_category_source
from index import IndexManager
from retrieval import RetrievalEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_FILES = 30

# Page configuration
st.set_page_config(
    page_title="DocInsight - Document Originality Analysis",
    page_icon="📄",
    layout="wide"
)


def display_originality_metrics(metrics: dict, doc_name: str = ""):
    """Display document-level originality metrics including plagiarism factor breakdown."""
    header = f"📊 Originality Analysis - {doc_name}" if doc_name else "📊 Originality Analysis"
    st.header(header)

    # Primary metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        originality_score = metrics.get('originality_score', 0.0)
        st.metric(label="Originality Score", value=f"{originality_score:.1%}")

    with col2:
        coverage = metrics.get('plagiarized_coverage', 0.0)
        st.metric(label="Plag. Coverage", value=f"{coverage:.1%}")

    with col3:
        severity = metrics.get('severity_index', 0.0)
        st.metric(label="Severity Index", value=f"{severity:.3f}")

    with col4:
        total_sentences = metrics.get('total_sentences', 0)
        st.metric(label="Sentences", value=total_sentences)

    with col5:
        plag_factor = metrics.get('plagiarism_factor', None)
        if plag_factor is not None:
            st.metric(label="Plagiarism Factor", value=f"{plag_factor:.3f}")
        else:
            st.metric(label="Plagiarism Factor", value="—")

    # Detailed breakdown
    if 'plagiarism_components' in metrics:
        with st.expander("🔬 Plagiarism Factor Breakdown"):
            comps = metrics['plagiarism_components']
            weights = comps.get('weights', {})
            st.write("The plagiarism factor combines weighted components: ")
            st.write(f"- Coverage Component: {comps.get('coverage_component',0.0):.4f}")
            st.write(f"- Severity Component: {comps.get('severity_component',0.0):.4f}")
            st.write(f"- Span Ratio Component: {comps.get('span_ratio_component',0.0):.4f}")
            st.write(f"Weights α={weights.get('alpha')} β={weights.get('beta')} γ={weights.get('gamma')}")
            st.caption("Originality Score = 1 - Plagiarism Factor (clamped ≥ 0). Lower factor → higher originality.")


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
            if st.checkbox(f"Show sentences in span {i}", key=f"span_{i}_{span['start_index']}"):
                st.write("**Sentences in this span:**")
                for j, sent_result in enumerate(span['sentences']):
                    sentence = sent_result.get('sentence', '')
                    confidence = sent_result.get('confidence_score', 0.0)
                    best_match = sent_result.get('best_match', '')
                    
                    st.write(f"**{j+1}.** {sentence}")
                    if best_match:
                        st.write(f"   *Similar to:* {best_match} (confidence: {confidence:.3f})")
                    st.write("---")


def display_sentence_details(sentence_results: list, max_display: int = MAX_SENTENCE_DISPLAY, doc_key: str = ""):
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
            index=0,
            key=f"risk_filter_{doc_key}"
        )
    
    with col2:
        show_details = st.checkbox("Show detailed scores", value=False, key=f"show_details_{doc_key}")
    
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
                st.write(f"**Confidence (fused):** {confidence:.3f}")
                if 'match_strength' in result:
                    st.write(f"**Match strength:** {result.get('match_strength')} ⟶ {result.get('reason','')}")
            
            if show_details:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"Semantic: {result.get('semantic_score', 0.0):.3f}")
                with col2:
                    st.write(f"Cross-encoder: {result.get('rerank_score', 0.0):.3f}")
                with col3:
                    st.write(f"Stylometry: {result.get('stylometry_score', 0.0):.3f}")
                # Component normalized contributions if available
                components = result.get('components', {})
                if components:
                    st.write(f"Components (normalized): Semantic={components.get('semantic',0):.2f}, Rerank={components.get('cross_encoder',0):.2f}, Styl={components.get('stylometry',0):.2f}")
            
            st.write("---")


def create_download_buttons(report_files: dict, doc_name: str):
    """Create download buttons for reports"""
    st.subheader(f"📥 Download Reports - {doc_name}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'html' in report_files and os.path.exists(report_files['html']):
            with open(report_files['html'], 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            st.download_button(
                label="📄 Download HTML Report",
                data=html_content,
                file_name=f"{Path(doc_name).stem}_report.html",
                mime="text/html",
                key=f"html_download_{doc_name}"
            )
    
    with col2:
        if 'json' in report_files and os.path.exists(report_files['json']):
            with open(report_files['json'], 'r', encoding='utf-8') as f:
                json_content = f.read()
            
            st.download_button(
                label="📊 Download JSON Report",
                data=json_content,
                file_name=f"{Path(doc_name).stem}_report.json",
                mime="application/json",
                key=f"json_download_{doc_name}"
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

        # Semantic model details
        sem_meta = processing_info.get('semantic_model') or {}
        if sem_meta:
            st.write("---")
            st.write("**Semantic Model:**")
            st.write(f"Source: {sem_meta.get('source','?')}")
            st.write(f"Path: {sem_meta.get('path','?')}")
            st.write(f"Fine-tuned flag: {sem_meta.get('use_fine_tuned_flag')}")


@st.cache_resource(show_spinner=False)
def get_cached_pipeline():
    """Create and cache the analysis pipeline (models + indexes)."""
    return DocumentAnalysisPipeline()


@st.cache_data(show_spinner=False)
def analyze_file_cached(temp_path: str, file_bytes_hash: str):
    """Cached analysis function"""
    pipeline = get_cached_pipeline()
    return pipeline.analyze_document(temp_path)


def display_batch_summary(results: Dict[str, dict]):
    """Display summary of all analyzed documents"""
    st.header("📊 Batch Analysis Summary")
    
    if not results:
        st.info("No documents analyzed yet.")
        return
    
    # Create summary dataframe
    summary_data = []
    for filename, result_data in results.items():
        try:
            # Extract the analysis result from the stored structure
            analysis_result = result_data.get('analysis', {})
            metrics = analysis_result.get('originality_analysis', {}).get('originality_metrics', {})
            
            originality_score = metrics.get('originality_score', 0.0)
            plag_coverage = metrics.get('plagiarized_coverage', 0.0)
            sentence_dist = metrics.get('sentence_distribution', {})
            high_risk = sentence_dist.get('HIGH', 0)
            total_sentences = metrics.get('total_sentences', 0)
            
            summary_data.append({
                'Document': filename,
                'Originality Score': f"{originality_score:.1%}",
                'Plagiarism Coverage': f"{plag_coverage:.1%}",
                'High Risk Sentences': high_risk,
                'Total Sentences': total_sentences
            })
        except Exception as e:
            logger.error(f"Error processing summary for {filename}: {e}")
            summary_data.append({
                'Document': filename,
                'Originality Score': "Error",
                'Plagiarism Coverage': "Error",
                'High Risk Sentences': "—",
                'Total Sentences': "—"
            })
    
    st.dataframe(summary_data, use_container_width=True)


def main():
    """Main Streamlit application with multi-file support"""
    st.title("📄 DocInsight - Document Originality Analysis")
    st.markdown(f"Upload up to **{MAX_FILES} documents** to analyze their originality and detect potential plagiarism.")
    
    # CHECK IF CORPUS IS READY
    corpus_ready = False
    try:
        dbm = get_db_manager()
        stats = dbm.get_corpus_stats()
        corpus_ready = stats.get('total_chunks', 0) > 0 or stats.get('embedded_chunks', 0) > 0
    except:
        corpus_ready = False
    
    if not corpus_ready:
        st.warning("⚠️ Academic corpus must be built first")
        
        if st.button("Build Academic Corpus Now"):
            pass
        
        st.stop()
    
    st.success("✅ Corpus ready")
    
    # Multi-file upload
    uploaded_files = st.file_uploader(
        f"Choose up to {MAX_FILES} documents to analyze",
        type=[ext.lstrip('.') for ext in SUPPORTED_EXTENSIONS],
        help=f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}",
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Limit number of files
        if len(uploaded_files) > MAX_FILES:
            st.error(f"❌ Please upload no more than {MAX_FILES} files. You uploaded {len(uploaded_files)} files.")
            st.stop()
        
        st.info(f"📂 {len(uploaded_files)} file(s) selected for analysis")
        
        # Analysis button
        if st.button("🚀 Analyze All Documents", type="primary"):
            # Initialize results storage
            if 'batch_results' not in st.session_state:
                st.session_state.batch_results = {}
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Analyze each file
            for idx, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name
                status_text.text(f"Analyzing {idx + 1}/{len(uploaded_files)}: {filename}")
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    temp_path = tmp_file.name
                
                try:
                    # Hash file bytes for caching
                    uploaded_file.seek(0)
                    import hashlib
                    file_bytes = uploaded_file.getvalue()
                    file_hash = hashlib.md5(file_bytes).hexdigest()
                    
                    # Analyze document
                    analysis_result = analyze_file_cached(temp_path, file_hash)
                    
                    # Generate reports
                    pipeline = get_cached_pipeline()
                    report_files = pipeline.generate_report_files(analysis_result)
                    
                    # Store results
                    st.session_state.batch_results[filename] = {
                        'analysis': analysis_result,
                        'reports': report_files
                    }
                    
                except Exception as e:
                    st.error(f"❌ Error analyzing {filename}: {str(e)}")
                    logger.error(f"Error analyzing {filename}: {e}")
                    logger.error(traceback.format_exc())
                
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        logger.warning(f"Could not delete temporary file {temp_path}: {e}")
                
                # Update progress
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("✅ All documents analyzed!")
            st.success(f"Successfully analyzed {len(st.session_state.batch_results)} document(s)")
    
    # Display results if available
    if 'batch_results' in st.session_state and st.session_state.batch_results:
        st.divider()
        
        # Batch summary
        display_batch_summary(st.session_state.batch_results)
        
        st.divider()
        
        # Individual document results
        st.header("📑 Individual Document Reports")
        
        # Document selector
        selected_doc = st.selectbox(
            "Select a document to view detailed results:",
            options=list(st.session_state.batch_results.keys())
        )
        
        if selected_doc:
            result_data = st.session_state.batch_results[selected_doc]
            analysis_result = result_data['analysis']
            report_files = result_data['reports']
            
            # Extract components
            originality_analysis = analysis_result.get('originality_analysis', {})
            originality_metrics = originality_analysis.get('originality_metrics', {})
            top_risk_spans = originality_analysis.get('top_risk_spans', [])
            sentence_results = analysis_result.get('sentence_results', [])
            sentence_distribution = originality_metrics.get('sentence_distribution', {})
            processing_info = analysis_result.get('processing_info', {})
            citation_info = analysis_result.get('citations', {})
            
            # Display metrics
            display_originality_metrics(originality_metrics, selected_doc)
            
            # Distribution and risk spans
            col1, col2 = st.columns([1, 2])
            with col1:
                display_sentence_distribution(sentence_distribution)
            with col2:
                display_top_risk_spans(top_risk_spans)
            
            # Sentence details
            display_sentence_details(sentence_results, doc_key=selected_doc)
            
            # Download buttons
            create_download_buttons(report_files, selected_doc)
            
            # Processing info
            display_processing_info(processing_info)
            
            # Citation summary
            if citation_info:
                with st.expander("📚 Citation Summary"):
                    st.write(f"Masking enabled: {citation_info.get('masking_enabled')}")
                    summary = citation_info.get('summary', {})
                    if summary:
                        st.write({k: v for k, v in summary.items() if k != 'total'})
                        st.write(f"Total citations masked: {summary.get('total',0)}")
    
    # Sidebar
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
        st.markdown(f"""
        - **Batch processing**: Up to {MAX_FILES} files at once
        - Support for PDF, DOCX, and TXT files
        - Real-time similarity detection
        - Detailed sentence-level analysis
        - Downloadable HTML/JSON reports
        - Risk span clustering
        - Processing component status
        """)
        
        st.header("📚 Corpus Management")
        st.caption("Build and manage your academic corpus for document analysis.")
        
        try:
            _dbm_flag = get_db_manager()
            starter_done = bool(_dbm_flag.get_setting('starter_corpus_built', False))
        except Exception:
            starter_done = False
        
        starter_label = "Academic Corpus Built ✔" if starter_done else "Build Academic Corpus (~20 docs)"
        starter_disabled = starter_done
        starter_btn = st.button(starter_label, disabled=starter_disabled, use_container_width=True)
        
        if starter_btn and not starter_done:
            try:
                with st.spinner("Building academic corpus across multiple domains..."):
                    dbm = get_db_manager()
                    pipeline = IngestionPipeline(dbm)
                    
                    wiki_topics = [
                        "Machine learning", "Climate change", "Photosynthesis", "French Revolution",
                        "Neural network", "Quantum computing", "Econometrics", "Genetics", 
                        "Cybersecurity", "Data structures"
                    ]
                    for topic in wiki_topics:
                        src = create_wiki_search_source(topic)
                        pipeline.ingest_source(src)
                    
                    arxiv_categories = ["cs.AI", "cs.CL", "stat.ML", "math.OC", "physics.comp-ph"]
                    for cat in arxiv_categories:
                        src = create_arxiv_category_source(cat, max_results=3)
                        pipeline.ingest_source(src)
                    
                    idx = IndexManager(dbm)
                    build_stats = idx.build_index(force_rebuild=False)
                    dbm.set_setting('starter_corpus_built', True, 'Prebuilt multi-domain academic corpus')
                
                st.success("Academic corpus ready and indexed ✔")
                st.write({
                    "index_type": build_stats.get("index_type"),
                    "chunks_indexed": build_stats.get("chunks_indexed", 0),
                    "embeddings_generated": build_stats.get("embeddings_generated", 0),
                    "build_successful": build_stats.get("build_successful", False)
                })
            except Exception as e:
                st.error(f"Academic corpus build failed: {e}")


if __name__ == "__main__":
    main()