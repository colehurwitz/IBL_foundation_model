#!/bin/bash
# 
conda activate ibl-fm

MODEL_NAME=${1} # Model name
MASK_MODE=${2} # Mask mode e.g. all,temporal
NUM_TRAIN_SESSIONS=${3} # Number of training sessions, e.g. 1, 10, 34
MODE=${4} # train,eval,train-eval

while IFS= read -r line
do
    echo "Evaluate on test eid: $line"
    sbatch finetune_eval_multi_session.sh $MODEL_NAME $MASK_MODE $NUM_TRAIN_SESSIONS $line $MODE
done < "../../data/test_re_eids.txt"

# conda deactivate