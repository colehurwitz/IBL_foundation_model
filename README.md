# International Brain Laboratory (IBL) Foundation Model

<p align="center">
    <img src=assets/figure_1.jpg />
</p>

## Environment setup

Create conda environment

```bash
conda env create -f env.yaml
```

Activate the environment

```bash
conda activate ibl-fm
```

## Datasets

**IBL datasets**

  

We included 100 sessions IBL dataset and upload to huggingface. Please look at the dataset [here](https://huggingface.co/neurofm123?message=You%27re%20already%20a%20member%20of%20neurofm123!), and ask a team member to invite you.

Please create a access token using to download the dataset. You can generate the token through huggingface User `settings/Access Token`, the type of token is Read.

Use `huggingface-cli login` to login. 

You are all set now.

## Training Multi/Single Session Models (SSL)

### Setup and Start Training
1. **Navigate to the script directory:**
   ```bash
   cd script
   ```

2. **Start the training process:**
   ```bash
   source train_sessions.sh
   ```

### Configuration Adjustments
- **Modify Model Configurations:**
  To change the model for training, update the YAML files in `src/configs` and adjust settings in `src/train_sessions.py`.

- **Example of Trainer and Mode Configurations:**
  ```python
  # Default setting
  # Load configuration
  kwargs = {
    "model": "include:src/configs/ndt1_stitching.yaml"
  }
  config = config_from_kwargs(kwargs)
  config = update_config("src/configs/ndt1_stitching.yaml", config)
  config = update_config("src/configs/ssl_sessions_trainer.yaml", config)
  ```

- **Setting the Number of Sessions:**
  To determine the number of sessions for training, edit `ssl_sessions_trainer.yaml`. The paper used configurations of 1, 10, or 34 sessions.
  ```yaml
  num_sessions: 10  # Number of sessions to use in SSL training.
  ```

- **Training Logs:**
  Training logs will be uploaded to Weights & Biases (wandb) and saved in the `results` folder.

## Fine-Tuning and Evaluating the Pre-trained Model

### Notes
The scripts provided are designed for use on a High-Performance Computing (HPC) environment with Slurm. They allow for fine-tuning and evaluation of the model using multiple test sessions.

### Running Multi-Session Fine-Tuning and Evaluation
1. **Script for Multiple Sessions:**
   To submit all test sessions listed in `data/test_re_eids.txt` for fine-tuning and evaluation, use the following command:
   ```bash
   source run_finetune_multi_session.sh NDT1 all 10 train-eval
   ```

### Running Single Test Session Fine-Tuning and Evaluation
1. **Script for a Single Session:**
   To execute fine-tuning and evaluation for a specific test session, use the command below. Replace the placeholder for EID with the actual unique ID of the test session.
   ```bash
   source finetune_eval_multi_session.sh NDT1 all 10 5dcee0eb-b34d-4652-acc3-d10afc6eae68 train-eval
   ```

### Parameters Explanation
- `MODEL_NAME`: The name of the model (e.g., NDT1, NDT2).
- `MASK_MODE`: The masking mode to apply (e.g., all, temporal).
- `NUM_TRAIN_SESSIONS`: Number of training sessions to be used (e.g., 1, 10, 34).
- `EID`: Unique identifier for a specific test session.
- `MODE`: The operation mode (e.g., train, eval, train-eval).

### Output
Both scripts load the pre-trained model from the `results` folder and save the evaluation results in `.npy` files.

## Reading Out Results

### Visualizing Results
1. **Navigate to the script directory:**
   ```bash
   cd script
   ```

2. **Run the visualization script:**
   ```bash
   source draw.sh NUM_TRAIN_SESSIONS
   ```

   This script outputs images visualizing results metrics, which are stored in the `results/table` folder.
  

## Models

**Neural Data Transformer (NDT1) - re-implementation**
  
The configuration for NDT1 is `src/configs/ndt1.yaml`. Set the number of neurons by:

```

n_channels: NUM_NEURON # number of neurons recorded

```
