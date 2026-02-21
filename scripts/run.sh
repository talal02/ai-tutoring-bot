#!/bin/bash
#BSUB -J ai_tutor_api
#BSUB -o logs/api_%J.out
#BSUB -e logs/api_%J.err
#BSUB -n 2
#BSUB -R "span[hosts=1]"
#BSUB -M 40000
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -q BatchGPU
#BSUB -W 24:00

# Set up environment
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export TORCH_HOME=$HOME/.cache/torch
export TOKENIZERS_PARALLELISM=false
# Compute nodes have no internet - load all models from local cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Create necessary directories
mkdir -p logs

# Print job information
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $LSB_JOBID"
echo "Working directory: $(pwd)"
echo "GPU info:"
nvidia-smi

NODE=$(hostname)
PORT=8000

echo ""
echo "============================================"
echo " To access the web UI, open a NEW terminal:"
echo "   ssh -L ${PORT}:${NODE}:${PORT} -N -o ServerAliveInterval=60 -o ServerAliveCountMax=10 taah5267@<login-node>"
echo " Then open: http://localhost:${PORT}"
echo " (Keep that terminal open while using the UI)"
echo "============================================"
echo ""

cd /home/taah5267/ai-tutoring-bot

echo "Starting API server..."
python -m api.main

echo "Server exited at: $(date)"
