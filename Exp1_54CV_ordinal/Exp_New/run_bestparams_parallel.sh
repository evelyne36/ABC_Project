#!/bin/bash
# Parallel launcher for exp1_bestparams.py
# Usage:
#   ./run_bestparams_parallel.sh all  # Run all datasets in parallel
#   ./run_bestparams_parallel.sh dataset1.csv dataset2.csv  # Run specific datasets in parallel

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
echo "Running exp1_bestparams.py (PARALLEL)"
echo "=========================================="
echo "Datasets: ${DATASETS[@]}"
echo "Outer K: $OUTER_K, Inner K: $INNER_K"
echo "Parallel jobs: ${#DATASETS[@]}"
echo "=========================================="
echo ""

START_TIME=$(date +%s)

# Arrays to track PIDs and dataset names
declare -a PIDS
declare -a DATASET_NAMES

# Launch all datasets in parallel
for dataset in "${DATASETS[@]}"; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching: $dataset"

    python exp1_bestparams.py \
        --outer_k $OUTER_K \
        --inner_k $INNER_K \
        --search $SEARCH \
        --dataset "$dataset" &

    PIDS+=($!)
    DATASET_NAMES+=("$dataset")
done

echo ""
echo "All ${#DATASETS[@]} jobs launched. Waiting for completion..."
echo ""

# Wait for all processes and track results
SUCCESS_COUNT=0
FAIL_COUNT=0

for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    dataset=${DATASET_NAMES[$i]}

    wait $pid
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Completed: $dataset"
        ((SUCCESS_COUNT++))
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ Failed: $dataset (exit code: $exit_code)"
        ((FAIL_COUNT++))
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo "All parallel jobs completed!"
echo "Success: $SUCCESS_COUNT, Failed: $FAIL_COUNT"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=========================================="
