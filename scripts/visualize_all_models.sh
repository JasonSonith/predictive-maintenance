#!/bin/bash
# Batch generate visualizations for all trained models

set -e  # Exit on error

echo "========================================"
echo "Batch Visualization Generator"
echo "========================================"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Define models and their corresponding data
declare -A MODELS=(
    # IMS models
    ["ims_iforest"]="artifacts/models/ims_iforest.joblib:data/features/ims/ims_features.csv:artifacts/scores/ims_iforest_scores.csv"
    ["ims_knn_lof"]="artifacts/models/ims_knn_lof.joblib:data/features/ims/ims_features.csv:artifacts/scores/ims_knn_lof_scores.csv"
    ["ims_ocsvm"]="artifacts/models/ims_ocsvm.joblib:data/features/ims/ims_features.csv:artifacts/scores/ims_ocsvm_scores.csv"
    ["ims_autoencoder"]="artifacts/models/ims_autoencoder.pth:data/features/ims/ims_features.csv:artifacts/scores/ims_autoencoder_scores.csv"

    # CWRU models
    ["cwru_iforest"]="artifacts/models/cwru_iforest.joblib:data/features/cwru/cwru_features.csv:artifacts/scores/cwru_iforest_scores.csv"

    # AI4I models
    ["ai4i_iforest"]="artifacts/models/ai4i_iforest.joblib:data/features/ai4i/ai4i_features.csv:artifacts/scores/ai4i_iforest_scores.csv"

    # NASA C-MAPSS models
    ["fd001_iforest"]="artifacts/models/fd001_iforest.joblib:data/features/cmapss/fd001_features.csv:artifacts/scores/fd001_iforest_scores.csv"
    ["fd002_iforest"]="artifacts/models/fd002_iforest.joblib:data/features/cmapss/fd002_features.csv:artifacts/scores/fd002_iforest_scores.csv"
    ["fd003_iforest"]="artifacts/models/fd003_iforest.joblib:data/features/cmapss/fd003_features.csv:artifacts/scores/fd003_iforest_scores.csv"
    ["fd004_iforest"]="artifacts/models/fd004_iforest.joblib:data/features/cmapss/fd004_features.csv:artifacts/scores/fd004_iforest_scores.csv"
)

# Counter for tracking
TOTAL=${#MODELS[@]}
SUCCESS=0
SKIPPED=0

# Loop through each model
for model_name in "${!MODELS[@]}"; do
    IFS=':' read -r model_path features_path scores_path <<< "${MODELS[$model_name]}"

    echo "Processing: $model_name"
    echo "  Model: $model_path"
    echo "  Features: $features_path"
    echo "  Scores: $scores_path"

    # Check if model exists
    if [ ! -f "$model_path" ]; then
        echo "  ⚠️  Model not found - SKIPPING"
        ((SKIPPED++))
        echo ""
        continue
    fi

    # Check if features exist
    if [ ! -f "$features_path" ]; then
        echo "  ⚠️  Features not found - SKIPPING"
        ((SKIPPED++))
        echo ""
        continue
    fi

    # Generate visualizations
    OUTPUT_DIR="artifacts/figures/$model_name"
    echo "  Generating visualizations to: $OUTPUT_DIR"

    python scripts/visualize_shap.py \
        --model "$model_path" \
        --features "$features_path" \
        --scores "$scores_path" \
        --output "$OUTPUT_DIR" \
        --sample-size 300 \
        2>&1 | grep -E "✅|⚠️|❌|Saved" || true

    if [ $? -eq 0 ]; then
        echo "  ✅ Success"
        ((SUCCESS++))
    else
        echo "  ❌ Failed"
    fi

    echo ""
done

echo "========================================"
echo "Batch Processing Complete"
echo "========================================"
echo "Total models: $TOTAL"
echo "Successful: $SUCCESS"
echo "Skipped: $SKIPPED"
echo ""
echo "All visualizations saved to: artifacts/figures/"
