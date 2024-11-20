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


TRAIN=False
EVAL=False

echo "Model name: $MODEL_NAME, Mask mode: $MASK_MODE, Num train sessions: $NUM_TRAIN_SESSIONS, Test eid: $TEST_EID"
echo "Prompting: $PROMPTING"

conda activate ibl-fm

cd ../

python src/finetune_eval_multi_session.py --mask_ratio 0.3 \
                         --model_name NDT1_with_region_stitcher \
                         --prompting False \
                         --train True \
                         --eval True\
                         --base_path /home/ywang74/Dev/IBL_foundation_model 

cd scripts

conda deactivate