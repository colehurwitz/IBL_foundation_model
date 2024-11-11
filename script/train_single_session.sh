#!/bin/bash
#SBATCH -A bcxj-delta-gpu
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpuA100x8
#SBATCH --gpus=1
#SBATCH --mem=128g
#SBATCH --job-name=hyper_search
#SBATCH --output=outputs/output_%A_%a.txt
#SBATCH --array=0

# Activate conda environment
source ~/.bashrc  # Ensure conda is properly set up in the shell environment
conda activate ibl-fm

python src/finetune_eval_multi_session.py 