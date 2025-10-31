#!/bin/bash
#SBATCH --job-name=earlybird_all
#SBATCH --output=logs/earlybird_all_%j.out
#SBATCH --error=logs/earlybird_all_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-6159

# --- Configuration ---
DATASETS=("devign" "bifi" "reveal")
COMBS=("w_sum_tokens_w_sum_layers" "max_pool_layers_w_sum_tokens" "w_sum_layers_w_sum_tokens" "max_pool_tokens_w_sum_layers" "max_pool_layers_max_pool_tokens" "w_sum_cls" "max_pool_cls")
ONE_LAYER_COMBS=("one_layer_cls" "one_layer_w_sum_tokens" "one_layer_max_pool_tokens")
SEEDS=(1 2 3 4 5 6 7 8 9 42)
LAYERS=(1 2 3 4 5 6 7 8 9 10 11 12)
CUTOFF_LAYERS=(1 2 3 4 5 6 7 8 9 10 11)
INDEX=0

# --- Run all combinations ---
# cutoff_layers_one_layer_cls combination
for DATASET in "${DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for LAYER in ${CUTOFF_LAYERS[@]}; do
      echo "Running dataset=$DATASET, comb=$COMB, seed=$SEED, layer=$LAYER"
      python -m src.run --config_path src/config.yaml --model_name codebert --model_path "checkpoints/reused/model/codebert-base" --tokenizer_path "checkpoints/reused/model/codebert-base" --dataset_name $DATASET --benchmark_name acc --train --test -warmup 0 --device cuda --epochs 10 -clf one_linear_layer --combination_type cutoff_layers_one_layer_cls --hidden_layer_to_use $LAYER --experiment_no $INDEX --seed $SEED
       INDEX=$((INDEX+1))
     done
  done
done

# one layer combinations
for DATASET in "${DATASETS[@]}"; do
  for COMB in "${ONE_LAYER_COMBS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      for LAYER in ${LAYERS[@]}; do
        echo "Running dataset=$DATASET, comb=$COMB, seed=$SEED, layer=$LAYER"
        python -m src.run --config_path src/config.yaml --model_name codebert --model_path "checkpoints/reused/model/codebert-base" --tokenizer_path "checkpoints/reused/model/codebert-base" --dataset_name $DATASET --benchmark_name acc --train --test -warmup 0 --device cuda --epochs 10 -clf one_linear_layer --combination_type $COMB --hidden_layer_to_use $LAYER --experiment_no $INDEX --seed $SEED
        INDEX=$((INDEX+1))
      done
    done
  done
done

# all other combinations
for DATASET in "${DATASETS[@]}"; do
  for COMB in "${COMBS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      echo "Running dataset=$DATASET, comb=$COMB, seed=$SEED, layer=$LAYER"
      python -m src.run --config_path src/config.yaml --model_name codebert --model_path "checkpoints/reused/model/codebert-base" --tokenizer_path "checkpoints/reused/model/codebert-base" --dataset_name $DATASET --benchmark_name acc --train --test -warmup 0 --device cuda --epochs 10 -clf one_linear_layer --combination_type $COMB --experiment_no $INDEX --seed $SEED
      INDEX=$((INDEX+1))
    done
  done
done

