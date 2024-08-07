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

MODEL_NAME=${1}
MASK_MODE=${2}
NUM_TRAIN_SESSIONS=${3}
TEST_EID=${4}
MODE=${5}

TRAIN=False
EVAL=False

if [ $MASK_MODE == "all" ]; then
    PROMPTING=True
else
    PROMPTING=False
fi

# if train in MODE
if [[ $MODE == *"train"* ]]; then
    echo "Training"
    TRAIN=True
fi

if [[ $MODE == *"eval"* ]]; then
    echo "Evaluating"
    EVAL=True
fi

echo "Model name: $MODEL_NAME, Mask mode: $MASK_MODE, Num train sessions: $NUM_TRAIN_SESSIONS, Test eid: $TEST_EID"
echo "Prompting: $PROMPTING"

conda activate ibl-fm

cd ../

python src/finetune_eval_multi_session.py --mask_ratio 0.2 \
                         --mask_mode $MASK_MODE \
                         --model_name $MODEL_NAME \
                         --prompting $PROMPTING \
                         --train $TRAIN \
                         --eval $EVAL \
                         --base_path $SCRATCH/IBL_foundation_model \
                         --num_train_sessions $NUM_TRAIN_SESSIONS \
                         --test_eid $TEST_EID \
                         --use_nlb \
                         --seed 42

cd script

conda deactivate