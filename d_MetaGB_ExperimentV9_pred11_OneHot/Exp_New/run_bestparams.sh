#!/bin/bash
# Quick launcher for exp1_bestparams.py
# Usage:
#   ./run_bestparams.sh airfoil_self_noise.csv
#   ./run_bestparams.sh airfoil_self_noise.csv diamonds.csv
#   ./run_bestparams.sh all  # Run all datasets

set -e

OUTER_K=5
INNER_K=4
SEARCH="grid"

# All available datasets
ALL_DATASETS=(
    "airfoil_self_noise.csv"
    "Another-Dataset-on-used-Fiat-500.csv"
    "concrete_compressive_strength.csv"
    "diamonds.csv"
    "Food_Delivery_Time.csv"
    "healthcare_insurance_expenses.csv"
    "houses.csv"
    "miami_housing.csv"
    "physiochemical_protein.csv"
    "QSAR_fish_toxicity.csv"
    "wine_quality.csv"
)

if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset1.csv> [dataset2.csv ...] or 'all'"
    echo ""
    echo "Available datasets:"
    for ds in "${ALL_DATASETS[@]}"; do
        echo "  - $ds"
    done
    exit 1
fi

# Check if user wants all datasets
if [ "$1" == "all" ]; then
    DATASETS=("${ALL_DATASETS[@]}")
else
    DATASETS=("$@")
fi

echo "=========================================="
echo "Running exp1_bestparams.py"
echo "=========================================="
echo "Datasets: ${DATASETS[@]}"
echo "Outer K: $OUTER_K, Inner K: $INNER_K"
echo "=========================================="
echo ""

START_TIME=$(date +%s)

for dataset in "${DATASETS[@]}"; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: $dataset"

    python exp1_bestparams.py \
        --outer_k $OUTER_K \
        --inner_k $INNER_K \
        --search $SEARCH \
        --dataset "$dataset"

    if [ $? -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Completed: $dataset"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ Failed: $dataset"
    fi
    echo ""
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo "=========================================="
echo "All tasks completed!"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=========================================="
