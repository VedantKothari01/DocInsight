#!/bin/bash
"""
AI-likeness training orchestration script for DocInsight

Orchestrates AI-likeness classifier training with synthetic data generation.
"""

set -e  # Exit on any error

echo "🤖 DocInsight AI-likeness Training Pipeline"
echo "=========================================="

# Configuration
DATA_DIR="fine_tuning/data"
PAIRS_FILE="$DATA_DIR/pairs.csv"
MODEL_TYPE=${DOCINSIGHT_AI_MODEL_TYPE:-logistic}
AI_MODEL_PATH="models/ai_likeness"

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

if ! check_package "sklearn"; then
    print_warning "scikit-learn not installed. Installing..."
    pip install scikit-learn
fi

if ! check_package "pandas"; then
    print_warning "pandas not installed. Installing..."
    pip install pandas
fi

print_success "Dependencies checked"

# Step 1: Check for training data
print_step "Step 1: Checking for training data..."

if [ ! -f "$PAIRS_FILE" ]; then
    print_warning "Training data not found. Generating..."
    python scripts/generate_synthetic_pairs.py
    
    if [ $? -ne 0 ]; then
        print_error "Failed to generate training data"
        exit 1
    fi
    
    print_success "Training data generated"
else
    print_success "Training data found"
    
    # Show data statistics
    if command -v wc >/dev/null 2>&1; then
        PAIR_COUNT=$(wc -l < "$PAIRS_FILE")
        print_step "Found $((PAIR_COUNT - 1)) training pairs"
    fi
fi

# Step 2: Train AI-likeness classifier
print_step "Step 2: Training AI-likeness classifier..."

echo "Training parameters:"
echo "  Model type: $MODEL_TYPE"
echo "  Training data: $PAIRS_FILE"
echo "  Output path: $AI_MODEL_PATH"

python fine_tuning/train_ai_likeness.py \
    --pairs "$PAIRS_FILE" \
    --model-type "$MODEL_TYPE"

if [ $? -ne 0 ]; then
    print_error "AI-likeness training failed"
    exit 1
fi

print_success "AI-likeness classifier training completed"

# Step 3: Verify model was created
print_step "Step 3: Verifying trained model..."

if [ -f "$AI_MODEL_PATH/ai_likeness_model.pkl" ]; then
    print_success "AI-likeness model verified at $AI_MODEL_PATH"
    
    # Show model info if available
    if [ -f "$AI_MODEL_PATH/metrics.json" ]; then
        print_step "Model performance metrics:"
        python -c "
import json
try:
    with open('$AI_MODEL_PATH/metrics.json', 'r') as f:
        metrics = json.load(f)
    print(f\"  Model type: {metrics.get('model_type', 'N/A')}\")
    print(f\"  Accuracy: {metrics.get('accuracy', 0):.3f}\")
    print(f\"  Precision: {metrics.get('precision', 0):.3f}\")
    print(f\"  Recall: {metrics.get('recall', 0):.3f}\")
    print(f\"  F1 Score: {metrics.get('f1_score', 0):.3f}\")
    print(f\"  Training samples: {metrics.get('training_samples', 'N/A')}\")
    print(f\"  Test samples: {metrics.get('test_samples', 'N/A')}\")
except Exception as e:
    print(f\"  Could not load metrics: {e}\")
"
    fi
    
    if [ -f "$AI_MODEL_PATH/feature_schema.json" ]; then
        print_step "Feature information:"
        python -c "
import json
try:
    with open('$AI_MODEL_PATH/feature_schema.json', 'r') as f:
        schema = json.load(f)
    print(f\"  Total features: {schema.get('feature_count', 'N/A')}\")
    print(f\"  Stylometric features: {schema.get('stylometric_features', 'N/A')}\")
    print(f\"  Embedding features: {schema.get('embedding_features', 'N/A')}\")
    print(f\"  Embedding model: {schema.get('embedding_model', 'N/A')}\")
except Exception as e:
    print(f\"  Could not load schema: {e}\")
"
    fi
else
    print_warning "AI-likeness model not found, check logs for issues"
fi

# Summary
echo ""
echo "🎉 AI-likeness training pipeline completed!"
echo ""
echo "📁 Generated files:"
echo "  Training data: $DATA_DIR/"
echo "  AI-likeness model: $AI_MODEL_PATH/"
echo ""
echo "💡 Next steps:"
echo "  1. Test the AI-likeness classifier with sample texts"
echo "  2. Update your application to use the trained classifier"
echo "  3. Run the complete analysis pipeline with unified scoring"
echo ""
echo "🔧 Model files:"
echo "  Classifier: $AI_MODEL_PATH/ai_likeness_model.pkl"
echo "  Schema: $AI_MODEL_PATH/feature_schema.json"
echo "  Metrics: $AI_MODEL_PATH/metrics.json"
echo ""