#!/bin/bash
#SBATCH --account=col169
#SBATCH --partition=gpu-shared
#SBATCH --job-name="train_eval"
#SBATCH --output="train_eval.%j.out"
#SBATCH -N 100
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --mem 190000
#SBATCH --gpus=1
#SBATCH -t 0-10
#SBATCH --export=ALL

module load gpu
module load slurm

. ~/.bashrc
echo $TMPDIR
conda activate ibl-fm

cd /home/yzhang39/IBL_foundation_model

echo "Start job :"`date`
echo $SLURM_ARRAY_TASK_ID

huggingface-cli login --token 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  --add-to-git-credential

python src/train_eval_100.py --mask_mode temporal --mask_ratio 0.3 --eid_idx $SLURM_ARRAY_TASK_ID --train

conda deactivate

echo "Stop job :"`date`