#!/bin/bash
#SBATCH --account=bcxj-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name="ndt1"
#SBATCH --output="ndt1.%j.out"
#SBATCH -N 1
#SBACTH --array=0
#SBATCH -c 8
#SBATCH --ntasks-per-node=1
#SBATCH --mem 100000
#SBATCH --gpus=1
#SBATCH -t 0-8
#SBATCH --export=ALL


. ~/.bashrc
echo $TMPDIR

conda activate ibl-fm

cd ../

wandb login 3ec2f93e55245d103845fc9e0ad8a81aa3d44f7c

huggingface-cli login --token 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  --add-to-git-credential

python src/train_eval_single_session.py --model_name NDT1 --mask_mode temporal --mask_ratio 0.3 --eid db4df448-e449-4a6f-a0e7-288711e7a75a --cont_target whisker-motion-energy --train --eval --base_path /scratch/bcxj/yzhang39 --seed 42 --overwrite --save_plot

conda deactivate

