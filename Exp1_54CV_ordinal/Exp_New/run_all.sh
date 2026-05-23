#!/bin/bash

# Initialize conda
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi
eval "$(conda shell.bash hook)"
conda activate llm

echo "=========================================="
echo "Starting all experiments in parallel..."
echo "=========================================="

mkdir -p ./results ./logs

# Experiment 1: Each dataset in parallel
echo "Starting Experiment 1..."
DATASETS=("airfoil_self_noise.csv" "Another-Dataset-on-used-Fiat-500.csv" "concrete_compressive_strength.csv" "diamonds.csv" "Food_Delivery_Time.csv" "healthcare_insurance_expenses.csv" "houses.csv" "miami_housing.csv" "physiochemical_protein.csv" "QSAR_fish_toxicity.csv" "wine_quality.csv")

for ds in "${DATASETS[@]}"; do
    echo "  Launching: $ds"
    nohup python exp1_real_datasets.py --dataset "$ds" > "./logs/exp1_${ds%.csv}.log" 2>&1 &
done

# Experiment 2A
echo "Starting Experiment 2A..."
nohup python exp2A_synthetic_nonlinear.py > "./logs/exp2A.log" 2>&1 &

# Experiment 2B
echo "Starting Experiment 2B..."
nohup python exp2B_synthetic_interaction.py > "./logs/exp2B.log" 2>&1 &

echo "All experiments launched. Waiting..."
wait

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
