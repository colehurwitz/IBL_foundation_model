#!/bin/bash
#SBATCH -A bcxj-delta-gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpuA100x8
#SBATCH --gpus=1
#SBATCH --mem=256g
#SBATCH --job-name=hyper_search
#SBATCH --output=outputs2/output_%A_%a.txt
#SBATCH --array=0-143%16   # 72 jobs in total, with 8 running simultaneously

# Set up error handling to continue on failure
trap 'echo "Task $SLURM_ARRAY_TASK_ID failed. Continuing with next task..."' ERR

# Load necessary modules if required (optional)
# module load cuda/11.2

# Activate conda environment
source ~/.bashrc  # Ensure conda is properly set up in the shell environment
conda activate ibl-fm

# Define hyperparameter values in bash arrays
# n_pre_layers=(2 4)
# n_layers=(2 4)
# small_hidden_size=(32 64 128)
# hidden_size=(128 256 512)
# inter_size=(128 256 512)
n_pre_layers=(1 2 3)
n_layers=(3 4)
small_hidden_size=(32 64)
hidden_size=(256 512)
inter_size=(256 1024)
dropout=(0.3 0.6 0.8)

# Calculate the total number of combinations
total_combinations=$((${#n_pre_layers[@]} * ${#n_layers[@]} * ${#small_hidden_size[@]} * ${#hidden_size[@]} * ${#inter_size[@]} * ${#dropout[@]}))

# Get the SLURM array task ID
task_id=$SLURM_ARRAY_TASK_ID

# Calculate the indices for each parameter based on the task_id
index_inter_size=$((task_id % ${#inter_size[@]}))
task_id=$((task_id / ${#inter_size[@]}))

index_hidden_size=$((task_id % ${#hidden_size[@]}))
task_id=$((task_id / ${#hidden_size[@]}))

index_small_hidden_size=$((task_id % ${#small_hidden_size[@]}))
task_id=$((task_id / ${#small_hidden_size[@]}))

index_n_layers=$((task_id % ${#n_layers[@]}))
task_id=$((task_id / ${#n_layers[@]}))

index_dropout=$((task_id % ${#dropout[@]}))
task_id=$((task_id / ${#dropout[@]}))

index_n_pre_layers=$((task_id % ${#n_pre_layers[@]}))

# Extract the parameter values for the current task
n_pre_layers_value=${n_pre_layers[$index_n_pre_layers]}
n_layers_value=${n_layers[$index_n_layers]}
small_hidden_size_value=${small_hidden_size[$index_small_hidden_size]}
hidden_size_value=${hidden_size[$index_hidden_size]}
inter_size_value=${inter_size[$index_inter_size]}
dropout_value=${dropout[$index_dropout]}

# Run your Python script with the selected hyperparameters
python src/finetune_eval_multi_session.py --n_pre_layers $n_pre_layers_value --n_layers $n_layers_value --small_hidden_size $small_hidden_size_value --hidden_size $hidden_size_value --inter_size $inter_size_value --train True --eval True --dropout $dropout_value

# Check for errors and continue
if [ $? -ne 0 ]; then
    echo "Error encountered in task $SLURM_ARRAY_TASK_ID. Logging and continuing."
else
    echo "Task $SLURM_ARRAY_TASK_ID completed successfully."
fi
