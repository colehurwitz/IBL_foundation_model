#!/bin/bash

#SBATCH -A bcxj-delta-gpu  # Account
#SBATCH --time=18:00:00    # Time limit
#SBATCH --nodes=1          # Number of nodes
#SBATCH --ntasks-per-node=1 # Number of tasks per node
#SBATCH --partition=gpuA100x8 # Partition
#SBATCH --gpus=1           # Number of GPUs
#SBATCH --mem=256g         # Memory allocation
#SBATCH --job-name=train_new_architecture  # Job name
#SBATCH --output=/u/csanthirasegaran/IBL_foundation_model/new_model_test/output_%j.log  # Standard output log (%j will be replaced with job ID)
#SBATCH --error=/u/csanthirasegaran/IBL_foundation_model/new_model_test/error_%j.log   # Standard error log (%j will be replaced with job ID)

# Activate conda environment
source /u/csanthirasegaran/miniconda3/etc/profile.d/conda.sh
conda activate ibl-fm

# Run your Python script
python /u/csanthirasegaran/IBL_foundation_model/src/finetune_eval_multi_session.py 

