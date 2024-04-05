  

# International Brain Laboratory (IBL) Foundation Model

  

## Environment setup

Create and activate conda environment

```

conda env create -f env.yaml

conda activate ibl-fm

```

  

## Datasets

**Lorenz dataset**

  

To use the lorenz dataset from the [NDT1](https://arxiv.org/abs/2108.01210) paper, please download the dataset from this [link](https://drive.google.com/file/d/1O5GxtX90uCgP9xlcmalHmVgC7DjNKO0j/view?usp=sharing).

In `src/configs/trainer.yaml`, set the following path and dataset names:

```

dataset_dir: PATH_TO_DATA_DIR

dataset_name: lorenz

```

**IBL datasets**

  

We included 100 sessions IBL dataset and upload to huggingface. Please look at the dataset [here](https://huggingface.co/neurofm123?message=You%27re%20already%20a%20member%20of%20neurofm123!), and ask a team member to invite you.

Please create a access token using to download the dataset. You can generate the token through huggingface User `settings/Access Token`, the type of token is Read.

Use `huggingface-cli login` to login. 

You are all set now.
  

## Models

**Neural Data Transformer (NDT1) - re-implementation**

  

The configuration for NDT1 is `src/configs/ndt1.yaml`. Set the number of neurons by:

```

n_channels: NUM_NEURON # number of neurons recorded

```

  

## Training

To train a model, first set the model config in `src/main.py`. For NDT1, set the config to:

```

kwargs = {

"model": "include:src/configs/ndt1.yaml"

}

```

Then, run the script inside of `script/hpc`:

```

source run.sh # Train model

```

  

## Evaluation

To evaluate a model, first set the pre-trained model path in `trainer.yaml`:

```

pretrained_model_path: PATH_TO_DIR/models/ndt1/last_model_lorenz.pth

```

Then, run the associated evaluation script in `script/hpc`. To run evaluation for NDT1 use:

```

source eval_ndt1.sh # Evaluate NDT1

```