#!/bin/bash

#SBATCH --account=col169
#SBATCH --partition=gpu-shared
#SBATCH --job-name="train"
#SBATCH --output="train.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --mem 150000
#SBATCH --gpus=1
#SBATCH -t 0-2:00
#SBATCH --export=ALL

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