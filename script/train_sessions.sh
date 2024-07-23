#!/bin/bash

#SBATCH --job-name=multi-session
#SBATCH --output=multi-session-%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:1
#SBATCH -t 3-12:00:00 
#SBATCH --mem=64g

. ~/.bashrc
conda activate ibl-fm

cd ..

python src/train_sessions.py

cd script

conda deactivate