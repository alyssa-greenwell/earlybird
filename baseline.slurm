#!/bin/bash
#SBATCH --job-name=earlybird_baseline
#SBATCH --output=logs/earlybird_baseline_%A_%a.out
#SBATCH --error=logs/earlybird_baseline_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --array=0-29

module purge
module load lang/Python/3.9.5-GCCcore-10.3.0
export TOKENIZERS_PARALLELISM=false
export MLFLOW_TRACKING_URI="file:$(pwd)/mlruns"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1
mkdir -p logs
source venv/bin/activate

DATASETS=("devign" "break_it_fix_it" "reveal")
SEEDS=(1 2 3 4 5 6 7 8 9 42)

# Build all combinations
CONFIGS=()

# all other combinations (no per-layer)
for DATASET in "${DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    CONFIGS+=("$DATASET $SEED")
  done
done

# Retrieve the config for this task
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
read -r DATASET SEED <<< "$CONFIG"

echo "Running dataset=$DATASET, seed=$SEED"

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
  --combination_type last_layer_cls \
  --seed $SEED \
  --experiment_no $SLURM_ARRAY_TASK_ID \
  --batch_size 16"


echo "$CMD"
eval "$CMD"
