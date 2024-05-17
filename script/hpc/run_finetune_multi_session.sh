#!/bin/bash

conda activate ibl-fm

MODEL_NAME=${1}
MASK_MODE=${2}
NUM_TRAIN_SESSIONS=${3}

while IFS= read -r line
do
    echo "Evaluate on test eid: $line"
    sbatch finetune_eval_multi_session.sh $MODEL_NAME $MASK_MODE $NUM_TRAIN_SESSIONS $line
done < "../../data/test_re_eids.txt"

# conda deactivate