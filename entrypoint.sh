#!/usr/bin/env bash
set -e

echo "🚀 Starting DocInsight container..."

# Prevent segmentation fault for multiprocessing
python3 -c "import multiprocessing; multiprocessing.set_start_method('forkserver', force=True)" || true

# Run setup automatically (skip if already ready)
if [ ! -f "corpus_cache/.docinsight_academic_ready" ]; then
    echo "⚙️ Running DocInsight setup..."
    python3 setup_docinsight.py --target-size 20000
else
    echo "✅ DocInsight setup already complete, skipping..."
fi

# Launch Streamlit web UI
echo "🌐 Launching Streamlit at http://0.0.0.0:8501"
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
