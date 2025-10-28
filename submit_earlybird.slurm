#!/bin/bash
#SBATCH --job-name=earlybird_all
#SBATCH --output=logs/earlybird_all_%j.out
#SBATCH --error=logs/earlybird_all_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Make sure MLflow uses a known tracking URI
export MLFLOW_TRACKING_URI="file:./mlruns"

# --- Configuration ---
DATASETS=("devign" "bifi" "reveal")
COMBS=("cutoff_layers_one_layer_cls" "one_layer_cls" "max_pool_cls" "w_sum_cls" "one_layer_max_pool_tokens" "max_pool_layers_max_pool_tokens" "max_pool_layers_w_sum_tokens" "max_pool_tokens_w_sum_layers" "one_layer_w_sum_tokens" "w_sum_tokens_w_sum_layers" "w_sum_layers_w_sum_tokens" "max_pool_tokens_max_pool_layers")
SEEDS=(1 2 3 4 5 6 7 8 9 10)
LAYERS=(1 2 3 4 5 6 7 8 9 10 11 12)

# --- Run all combinations ---
for DATASET in "${DATASETS[@]}"; do
  for COMB in "${COMBS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      for LAYER in ${LAYERS}; do
        echo "Running dataset=$DATASET, comb=$COMB, seed=$SEED, layer=$LAYER"
        python -m src.run --config_path src/config.yaml --model_name codebert --model_path "checkpoints/reused/model/codebert-base" --tokenizer_path "checkpoints/reused/model/codebert-base" --dataset_name $DATASET --benchmark_name acc --train --test -warmup 0 --device cuda --epochs 10 -clf one_linear_layer --combination_type $COMB --hidden_layer_to_use $LAYER --seed $SEED
      done
    done
  done
done

# --- Export MLflow runs to CSVs ---
mkdir -p ../output/tables
for DATASET in "${DATASETS[@]}"; do
  echo "Exporting MLflow results for $DATASET..."
  mlflow search --experiment-names $DATASET -o csv > ../output/tables/mlflow_${DATASET}.csv
done
