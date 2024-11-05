#!/bin/bash
#SBATCH -A bcxj-delta-gpu          # Account name
#SBATCH --time=1:00:00             # Time limit
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --partition=gpuA100x4       # Partition name
#SBATCH --gpus=1                    # Number of GPUs
#SBATCH --mem=128g                  # Memory allocation
#SBATCH --job-name=test_mem_run     # Job name
#SBATCH --output=outputs/output_%A_%a.txt  # Output file
#SBATCH --array=0                    # Job array index

# Activate conda environment
source ~/.bashrc                     # Ensure conda is properly set up in the shell environment
conda activate ibl-fm                # Activate your conda environment

# Run the Python script
python test_mem.py        # Adjust the path if necessary