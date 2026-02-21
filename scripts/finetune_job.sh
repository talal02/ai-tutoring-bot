#!/bin/bash
#BSUB -J finetune_llama_8b
#BSUB -o logs/finetune_8b_%J.out
#BSUB -e logs/finetune_8b_%J.err
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -M 48000
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -q BatchGPU
#BSUB -W 24:00

# Set up environment
# Do NOT set CUDA_VISIBLE_DEVICES manually - LSF sets it to the exclusive GPU it assigns
export HF_HOME=$HOME/.cache/huggingface
export TORCH_HOME=$HOME/.cache/torch
export OMP_NUM_THREADS=4
# Compute nodes have no internet - load all models from local cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Create necessary directories
mkdir -p logs
mkdir -p models/finetuned_8b

# Print job information
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $LSB_JOBID"
echo "Working directory: $(pwd)"
echo "CPU info: $(nproc) cores"
echo "GPU info:"
nvidia-smi

cd /home/taah5267/ai-tutoring-bot

# Run data preparation
echo "Preparing training data..."
python scripts/prepare_data.py

# Run fine-tuning (bf16 + LoRA, no quantization needed for 8B on A100 40GB)
echo "Starting fine-tuning..."
python scripts/finetune.py

echo "Job finished at: $(date)"
