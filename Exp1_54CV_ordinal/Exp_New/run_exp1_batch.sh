#!/bin/bash
# Batch script to run Experiment 1 on all TabArena datasets
# Usage:
#   ./run_exp1_batch.sh bestparams  # Run best params version
#   ./run_exp1_batch.sh ensembling  # Run ensembling version
#   ./run_exp1_batch.sh both        # Run both versions
set -e  # Exit on error
# Configuration
OUTER_K=5
INNER_K=4
SEARCH_METHOD="grid"
# Get experiment type from argument
EXP_TYPE=${1:-"bestparams"}
# Dataset list (from hyperparams.py EXP1_DATASETS)
DATASETS=(
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
echo "=========================================="
echo "Experiment 1 Batch Runner"
echo "=========================================="
echo "Experiment type: $EXP_TYPE"
echo "Outer K: $OUTER_K"
echo "Inner K: $INNER_K"
echo "Search method: $SEARCH_METHOD"
echo "Total datasets: ${#DATASETS[@]}"
echo "=========================================="
echo ""
# Function to run bestparams experiment
run_bestparams(){
    local dataset=$1
    local dataset_name="${dataset%.csv}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting bestparams for $dataset_name"
    python exp1_bestparams.py \
        --outer_k $OUTER_K \
        --inner_k $INNER_K \
        --search $SEARCH_METHOD \
        --dataset "$dataset"
    if [ $? -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Completed bestparams for $dataset_name"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ Failed bestparams for $dataset_name"
    fi
    echo ""
}
# Function to run ensembling experiment
run_ensembling() {
    local dataset=$1
    local dataset_name="${dataset%.csv}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ensembling for $dataset_name"
    python exp1_ensembling.py \
        --outer_k $OUTER_K \
        --inner_k $INNER_K \
        --search $SEARCH_METHOD \
        --dataset "$dataset"
    if [ $? -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Completed ensembling for $dataset_name"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ Failed ensembling for $dataset_name"
    fi
    echo ""
}
# Main execution
START_TIME=$(date +%s)
case $EXP_TYPE in
    "bestparams")
        echo "Running BESTPARAMS experiments..."
        for dataset in "${DATASETS[@]}"; do
            run_bestparams "$dataset"
        done
        ;;
    "ensembling")
        echo "Running ENSEMBLING experiments..."
        for dataset in "${DATASETS[@]}"; do
            run_ensembling "$dataset"
        done
        ;;
    "both")
        echo "Running BOTH experiments..."
        for dataset in "${DATASETS[@]}"; do
            run_bestparams "$dataset"
            run_ensembling "$dataset"
        done
        ;;
    *)
        echo "Error: Invalid experiment type '$EXP_TYPE'"
        echo "Usage: $0 {bestparams|ensembling|both}"
        exit 1
        ;;
esac
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))
echo "=========================================="
echo "Batch execution completed!"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=========================================="