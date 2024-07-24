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

# python src/train_lfp_ap.py --eid 5dcee0eb-b34d-4652-acc3-d10afc6eae68 --modality lfp --task ssl --train --base_path /scratch/bcxj/yzhang39

# python src/train_lfp_ap.py --eid 5dcee0eb-b34d-4652-acc3-d10afc6eae68 --modality lfp --task sl --train --base_path /scratch/bcxj/yzhang39

python src/train_lfp_ap.py --eid 5dcee0eb-b34d-4652-acc3-d10afc6eae68 --modality ap --task sl --train --base_path /scratch/bcxj/yzhang39

conda deactivate

