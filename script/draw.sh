#!/bin/bash

#SBATCH --job-name=ibl-fm
#SBATCH --output=ibl-fm.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 2:00:00 
#SBATCH --mem=64g

. ~/.bashrc
conda activate ibl-fm

cd ../

python src/draw_table.py

cd script

conda deactivate