# DocInsight Installation Troubleshooting Guide

This guide helps resolve common installation issues with DocInsight.

## Quick Fix for Python 3.12 Compatibility

The main issue has been resolved in the latest version. The original error:
```
ERROR: Exception:
...
pip._vendor.pyproject_hooks._impl.BackendUnavailable: Cannot import 'setuptools.build_meta'
```

**Has been fixed by:**
1. Updating requirements.txt to use Python 3.12 compatible versions
2. Removing restrictive version constraints that forced source builds
3. Improving the installation script with better error handling

## Installation Options

### Option 1: One-Command Setup (Recommended)
```bash
git clone https://github.com/VedantKothari01/DocInsight.git
cd DocInsight
bash run_docinsight.sh
```

### Option 2: Manual Installation (If network issues persist)
```bash
# Upgrade build tools first
pip install --upgrade setuptools wheel pip

# Install core dependencies
pip install streamlit numpy pandas nltk

# Install ML dependencies (may take longer)
pip install sentence-transformers transformers torch
pip install spacy faiss-cpu scikit-learn

# Install text processing tools
pip install textstat docx2txt PyMuPDF beautifulsoup4

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the application
streamlit run streamlit_app.py
```

### Option 3: Minimal Installation for Testing
```bash
# Install only essential packages for basic functionality
pip install streamlit numpy pandas nltk textstat docx2txt PyMuPDF

# Run basic tests
python test_core_functionality.py
```

## Troubleshooting Common Issues

### 1. Network Timeouts
If you see "Read timed out" errors:
- Try installing packages individually
- Use: `pip install --timeout 300 --retries 3 <package>`
- Check your internet connection

### 2. Build Failures
If packages try to build from source:
- Ensure you have the latest requirements.txt (no version constraints like `<1.25.0`)
- Upgrade pip: `pip install --upgrade pip setuptools wheel`

### 3. Python Version Issues
- Ensure you have Python 3.8 or higher
- For Python 3.12+, the updated requirements.txt should work without issues

### 4. Missing spaCy Model
If spaCy model download fails:
```bash
python -m spacy download en_core_web_sm
```

## Verifying the Fix

Run this to verify the original error is resolved:
```bash
python demo_fix_validation.py
```

You should see all tests pass, confirming the setuptools.build_meta error is fixed.