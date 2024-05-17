#!/bin/bash

#SBATCH --job-name=finetune-eval-multi
#SBATCH --output=finetune-eval-multi-%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:1
#SBATCH -t 2-12:00:00 
#SBATCH --mem=64g

MODEL_NAME=${1}
MASK_MODE=${2}
NUM_TRAIN_SESSIONS=${3}
TEST_EID=${4}

if [ $MASK_MODE == "all" ]; then
    PROMPTING=True
else
    PROMPTING=False
fi

echo "Model name: $MODEL_NAME, Mask mode: $MASK_MODE, Num train sessions: $NUM_TRAIN_SESSIONS, Test eid: $TEST_EID"
echo "Prompting: $PROMPTING"

conda activate ibl-fm

cd ../../

python src/finetune_eval_multi_session.py --mask_ratio 0.3 \
                         --mask_mode $MASK_MODE \
                         --model_name $MODEL_NAME \
                         --prompting $PROMPTING \
                         --train \
                         --eval \
                         --base_path $SCRATCH/IBL_foundation_model \
                         --num_train_sessions $NUM_TRAIN_SESSIONS \
                         --test_eid $TEST_EID

cd script/hpc

conda deactivate