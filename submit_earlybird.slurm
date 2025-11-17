#!/bin/bash
#SBATCH --job-name=earlybird_all
#SBATCH --output=logs/earlybird_all_%A_%a.out
#SBATCH --error=logs/earlybird_all_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-1619

module purge
module load lang/Python/3.9.5-GCCcore-10.3.0
export TOKENIZERS_PARALLELISM=false
export MLFLOW_TRACKING_URI="file:$(pwd)/mlruns"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1
mkdir -p logs
source venv/bin/activate

DATASETS=("devign" "bifi" "reveal")
COMBS=("w_sum_tokens_w_sum_layers" "max_pool_layers_w_sum_tokens" "w_sum_layers_w_sum_tokens" "max_pool_tokens_w_sum_layers" "max_pool_layers_max_pool_tokens" "w_sum_cls" "max_pool_cls")
ONE_LAYER_COMBS=("one_layer_cls" "one_layer_w_sum_tokens" "one_layer_max_pool_tokens")
SEEDS=(1 2 3 4 5 6 7 8 9 42)
LAYERS=(1 2 3 4 5 6 7 8 9 10 11 12)
CUTOFF_LAYERS=(1 2 3 4 5 6 7 8 9 10 11)

# Build all combinations
CONFIGS=()

# cutoff_layers_one_layer_cls combinations
for DATASET in "${DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for LAYER in "${CUTOFF_LAYERS[@]}"; do
      CONFIGS+=("$DATASET cutoff_layers_one_layer_cls $SEED $LAYER")
    done
  done
done

# one_layer combinations
for DATASET in "${DATASETS[@]}"; do
  for COMB in "${ONE_LAYER_COMBS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      for LAYER in "${LAYERS[@]}"; do
        CONFIGS+=("$DATASET $COMB $SEED $LAYER")
      done
    done
  done
done

# all other combinations (no per-layer)
for DATASET in "${DATASETS[@]}"; do
  for COMB in "${COMBS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      CONFIGS+=("$DATASET $COMB $SEED NA")
    done
  done
done

# Retrieve the config for this task
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
read -r DATASET COMB SEED LAYER <<< "$CONFIG"

echo "Running dataset=$DATASET, comb=$COMB, seed=$SEED, layer=$LAYER"

CMD="python -m src.run \
  --config_path src/config.yaml \
  --model_name codebert \
  --model_path checkpoints/reused/model/codebert-base \
  --tokenizer_path checkpoints/reused/model/codebert-base \
  --dataset_name $DATASET \
  --benchmark_name acc \
  --train --test \
  -warmup 0 \
  --device cuda \
  --epochs 10 \
  -clf one_linear_layer \
  --combination_type $COMB \
  --seed $SEED \
  --experiment_no $SLURM_ARRAY_TASK_ID \
  --batch_size 16"

if [ "$LAYER" != "NA" ]; then
  CMD="$CMD --hidden_layer_to_use $LAYER"
fi

echo "$CMD"
eval "$CMD"
