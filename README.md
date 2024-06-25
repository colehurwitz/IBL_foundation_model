  

# International Brain Laboratory (IBL) Foundation Model

<p align="center">
    <img src=assets/figure_1.jpg />
</p>

## Environment setup

Create and activate conda environment

```

conda env create -f env.yaml

conda activate ibl-fm

```

  

## Datasets

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