#!/bin/bash

#SBATCH --job-name=ibl-fm
#SBATCH --output=ibl-fm.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 2-12:00:00 
#SBATCH --mem=64g

conda activate ibl-fm

cd ../../

python src/train_eval.py --mask_ratio 0.3 \
                         --mask_mode all \
                         --model_name NDT1 \
                         --prompting \
                         --stitching \
                         --base_path $SCRATCH/IBL_foundation_model \
                         --eid ff96bfe1-d925-4553-94b5-bf8297adf259

cd script/hpc

conda deactivate