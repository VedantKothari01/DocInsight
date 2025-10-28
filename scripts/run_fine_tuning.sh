#!/bin/bash
"""
Fine-tuning orchestration script for DocInsight

Orchestrates the complete fine-tuning pipeline: dataset generation,
preparation, and semantic model fine-tuning.
"""

set -e  # Exit on any error

echo "🔧 DocInsight Fine-tuning Pipeline"
echo "=================================="

# Configuration
DATA_DIR="fine_tuning/data"
PAIRS_FILE="$DATA_DIR/pairs.csv"
EPOCHS=${DOCINSIGHT_FINE_TUNING_EPOCHS:-3}
BATCH_SIZE=${DOCINSIGHT_FINE_TUNING_BATCH_SIZE:-16}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if Python package is installed
check_package() {
    python -c "import $1" 2>/dev/null
    return $?
}

# Check dependencies
print_step "Checking dependencies..."

if ! check_package "sentence_transformers"; then
    print_warning "sentence-transformers not installed. Installing..."
    pip install sentence-transformers
fi

if ! check_package "pandas"; then
    print_warning "pandas not installed. Installing..."
    pip install pandas
fi

print_success "Dependencies checked"

# Step 1: Generate synthetic pairs if not exists
print_step "Step 1: Checking for training data..."

if [ ! -f "$PAIRS_FILE" ]; then
    print_warning "Synthetic pairs not found. Generating..."
    python scripts/generate_synthetic_pairs.py
    
    if [ $? -ne 0 ]; then
        print_error "Failed to generate synthetic pairs"
        exit 1
    fi
    
    print_success "Synthetic pairs generated"
else
    print_success "Synthetic pairs already exist"
    
    # Show pair statistics
    if command -v wc >/dev/null 2>&1; then
        PAIR_COUNT=$(wc -l < "$PAIRS_FILE")
        print_step "Found $((PAIR_COUNT - 1)) training pairs"
    fi
fi

# Step 2: Prepare dataset splits
print_step "Step 2: Preparing dataset splits..."

python -c "
from fine_tuning.dataset_prep import prepare_training_data
import sys

try:
    files = prepare_training_data('$PAIRS_FILE')
    print('Dataset preparation completed successfully')
    for name, path in files.items():
        print(f'  {name}: {path}')
except Exception as e:
    print(f'Dataset preparation failed: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    print_error "Dataset preparation failed"
    exit 1
fi

print_success "Dataset splits prepared"

# Step 3: Fine-tune semantic model
print_step "Step 3: Fine-tuning semantic model..."

echo "Training parameters:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Data directory: $DATA_DIR"

python fine_tuning/fine_tune_semantic.py \
    --data "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE"

if [ $? -ne 0 ]; then
    print_error "Fine-tuning failed"
    exit 1
fi

print_success "Semantic model fine-tuning completed"

# Step 4: Verify model was created
print_step "Step 4: Verifying fine-tuned model..."

MODEL_PATH="models/semantic_local"
if [ -f "$MODEL_PATH/config.json" ]; then
    print_success "Fine-tuned model verified at $MODEL_PATH"
    
    # Show model info if available
    if [ -f "$MODEL_PATH/training_info.json" ]; then
        print_step "Training information:"
        python -c "
import json
try:
    with open('$MODEL_PATH/training_info.json', 'r') as f:
        info = json.load(f)
    print(f\"  Base model: {info.get('base_model', 'N/A')}\")
    print(f\"  Training examples: {info.get('training_examples', 'N/A')}\")
    print(f\"  Status: {info.get('status', 'N/A')}\")
    if 'test_score' in info:
        print(f\"  Test score: {info['test_score']:.4f}\")
except Exception:
    pass
"
    fi
else
    print_warning "Fine-tuned model not found, check logs for issues"
fi

# Summary
echo ""
echo "🎉 Fine-tuning pipeline completed!"
echo ""
echo "📁 Generated files:"
echo "  Training data: $DATA_DIR/"
echo "  Fine-tuned model: $MODEL_PATH/"
echo ""
echo "💡 Next steps:"
echo "  1. Test the fine-tuned model with the main application"
echo "  2. Run AI-likeness training: bash scripts/run_ai_likeness_training.sh"
echo "  3. Update your Streamlit app to use the fine-tuned model"
echo ""
echo "🔧 Configuration:"
echo "  Set DOCINSIGHT_MODEL_FINE_TUNED_PATH=$MODEL_PATH to use fine-tuned model"
echo ""