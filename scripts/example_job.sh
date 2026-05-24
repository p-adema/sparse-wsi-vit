#!/bin/bash
#SBATCH --job-name=sparse-wsi-vit
#SBATCH --partition=gpu_a100              # use 'gpu_h100' for H100 nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Load modules
module purge
module load 2025
module load Anaconda3/2025.06-1

# Activate the project environment
cd $HOME/sparse-wsi-vit
source .venv/bin/activate           # created by 'uv sync'

# Pick a config
CONFIG=src/sparse_wsi_vit/configs/baseline_vit5dense_preembed_camelyon.py

uv run experiments/run.py --config $CONFIG