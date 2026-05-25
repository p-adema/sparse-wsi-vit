#!/bin/bash
#SBATCH --job-name=sparse-wsi-vit
#SBATCH --partition=gpu_a100              # use 'gpu_h100' for H100 nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mem=120G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err


# Pick a config,
CONFIG=src/sparse_wsi_vit/configs/baseline_vit5dense_preembed_camelyon.py

# and run!
uv run experiments/run.py --config $CONFIG