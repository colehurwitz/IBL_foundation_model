
# IBL Foundation Model  

## Environment setup

Create and activate conda environment

```

conda env create -f env.yaml

conda activate ibl-fm

```

## Dataset
1.  To use lorenz dataset(NDT1 paper). Plz download the dataset from [here](https://drive.google.com/file/d/1O5GxtX90uCgP9xlcmalHmVgC7DjNKO0j/view?usp=sharing). 
	In trainer.yaml:
	```
	dataset_dir: YOUR_PATH
	dataset_name: lorenz
	```
  In ndt1.yaml:
  ```
  n_channels: NUM_NEURON  # number of neurons recorded
  ``` 
	
2. The IBL dataset has been setup in huggingface.

  

## How to run

Run the script

```

source run.sh # Train model

```

## Evaluation

```
eval_ndt1.sh # Evaluate NDT1 SSL results
```

Setup the pre-trained model path in trainer.yaml:

```
pretrained_model_path: YOUR_PATH_TO/models/ndt1/last_model_lorenz.pth
```