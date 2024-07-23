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

NUM_TRAIN_SESSIONS=${1}

# default model is NDT1
# path to the data is in IBL_foundation_models/results

python src/draw_table.py --model NDT1 \
                         --base_path ./ \
                         --num_train_sessions ${NUM_TRAIN_SESSIONS} 


cd script

conda deactivate