from datasets import load_dataset, load_from_disk, concatenate_datasets
from accelerate import Accelerator
from loader.make_loader import make_loader, make_ndt2_loader
from utils.utils import set_seed,move_batch_to_device, viz_single_cell
from utils.config_utils import config_from_kwargs, update_config
from utils.dataset_utils import get_data_from_h5
from models.ndt1 import NDT1
from models.ndt2 import NDT2
import torch
import numpy as np
import os
from trainer.make import make_trainer

# load config
kwargs = {
    "model": "include:src/configs/ndt2.yaml"
}

config = config_from_kwargs(kwargs)
config = update_config("src/configs/ndt2.yaml", config)
config = update_config("src/configs/trainer.yaml", config)

# make log dir
log_dir = os.path.join(config.dirs.log_dir, "eval", "model_{}".format(config.model.model_class), "method_{}".format(config.method.model_kwargs.method_name))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# wandb
if config.wandb.use:
    import wandb
    wandb.init(project=config.wandb.project, entity=config.wandb.entity, config=config, name="eval_model_{}_method_{}".format(config.model.model_class, config.method.model_kwargs.method_name))

# set seed for reproducibility
set_seed(config.seed)

# download dataset from huggingface
if "ibl" in config.data.dataset_name:
    dataset = load_dataset(config.dirs.dataset_dir, cache_dir=config.dirs.dataset_cache_dir)
    # show the columns
    
    bin_size = dataset["train"]["bin_size"][0]
    

    # split the dataset to train and test
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=config.seed)
    # select the train dataset and the spikes_sparse_data column
    data_columns = ['spikes_sparse_data', 'spikes_sparse_indices', 'spikes_sparse_indptr', 'spikes_sparse_shape']
    train_dataset = dataset["train"].select_columns(data_columns)
    test_dataset = dataset["test"].select_columns(data_columns)

    if config.data.include_behav:
        dataset = load_from_disk(os.path.join('data', config.dirs.behav_dir))
        dataset = concatenate_datasets([dataset["train"], dataset["val"], dataset["test"]])
        dataset = dataset.train_test_split(test_size=0.1, seed=config.seed)
        bin_size = dataset["train"]["binsize"][0]

        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

    print(dataset.column_names)
    print(f"bin_size: {bin_size}")

else:
    train_dataset = get_data_from_h5("train", config.dirs.dataset_dir, config=config)
    test_dataset = get_data_from_h5("val", config.dirs.dataset_dir, config=config)
    bin_size = None

# make the dataloader
train_dataloader = make_loader(train_dataset, 
                         batch_size=config.training.train_batch_size, 
                         pad_to_right=True, 
                         patching=config.data.patching,
                         pad_value=-1.,
                         bin_size=bin_size,
                         max_time_length=config.data.max_time_length,
                         max_space_length=config.data.max_space_length,
                         n_neurons_per_patch=config.data.n_neurons_per_patch,
                         dataset_name=config.data.dataset_name,
                         shuffle=True)

test_dataloader = make_loader(test_dataset, 
                         batch_size=config.training.test_batch_size, 
                         pad_to_right=True, 
                         patching=config.data.patching,
                         pad_value=-1.,
                         bin_size=bin_size,
                         max_time_length=config.data.max_time_length,
                         max_space_length=config.data.max_space_length,
                         n_neurons_per_patch=config.data.n_neurons_per_patch,
                         dataset_name=config.data.dataset_name,
                         shuffle=False)

# Initialize the accelerator
accelerator = Accelerator()

# load model
NAME2MODEL = {"NDT1": NDT1, "NDT2": NDT2}
model_class = NAME2MODEL[config.model.model_class]
model = model_class(config.model, **config.method.model_kwargs)
# load pretrained model
model_ckpt = torch.load(os.path.join(config.dirs.pretrained_model_path))
model.load_state_dict(model_ckpt['model'].state_dict())
model = accelerator.prepare(model)

model.eval()
gt = []
preds = []
with torch.no_grad():
    for batch in test_dataloader:
        batch = move_batch_to_device(batch, accelerator.device)
        if config.data.patching:
            gt.append(batch['neuron_patches'].clone().reshape((-1, config.data.max_time_length, config.data.max_space_length*config.data.n_neurons_per_patch)))
        else:
            gt.append(batch['spikes_data'].clone())
        if config.data.patching:
            outputs = model(
                batch['neuron_patches'].flatten(1,-2), 
                batch['space_attention_mask'].flatten(1), 
                batch['time_attention_mask'].flatten(1),
                batch['spikes_spacestamps'].flatten(1),
                batch['spikes_timestamps'].flatten(1)
            )
            preds.append(outputs.preds.clone().reshape((-1, config.data.max_time_length, config.data.max_space_length*config.data.n_neurons_per_patch)))
        else:
            outputs = model(batch['spikes_data'], 
                              batch['attention_mask'], 
                              batch['spikes_timestamps'])
            preds.append(outputs.preds.clone())

gt = torch.cat(gt, dim=0)
preds = torch.cat(preds, dim=0)

if config.method.model_kwargs.loss == "poisson_nll":
    preds = torch.exp(preds)

# prepare the condition matrix
b_list = []

# choice
choice = np.array(test_dataset['choice'])
choice = np.tile(np.reshape(choice, (choice.shape[0], 1)), (1, 100))
b_list.append(choice)

# reward
reward = np.array(test_dataset['reward'])
reward = np.tile(np.reshape(reward, (reward.shape[0], 1)), (1, 100))
b_list.append(reward)

# block
block = np.array(test_dataset['block'])
block = np.tile(np.reshape(block, (block.shape[0], 1)), (1, 100))
b_list.append(block)

# wheel
wheel = np.array(test_dataset['wheel-speed'])
b_list.append(wheel)

behavior_set = np.stack(b_list,axis=-1)
print(behavior_set.shape)

# Settings for validation
X = behavior_set # [#trials, #timesteps, #variables]
ys = gt.cpu().numpy() # [#trials, #timesteps, #neurons]
y_preds = preds.cpu().numpy() # [#trials, #timesteps, #neurons]

var_name2idx = {'block':[2], 
                'choice': [0], 
                'reward': [1], 
                'wheel': [3],
                }

var_value2label = {'block': {(0.2,): "p(left)=0.2",
                            (0.5,): "p(left)=0.5",
                            (0.8,): "p(left)=0.8",},
                   'choice': {(-1.0,): "right",
                            (1.0,): "left"},
                   'reward': {(0.,): "no reward",
                            (1.,): "reward", } }

var_tasklist = ['block','choice','reward']
var_behlist = []

# choose more active neuron
tmp = np.mean(ys, axis=(0,1))
idx_top = np.argsort(tmp)[-5:]
print(idx_top)

import matplotlib.pyplot as plt

for i in range(idx_top.shape[0]):
    viz_single_cell(X,ys[:,:,idx_top[i]],y_preds[:,:,idx_top[i]], 
                    var_name2idx, var_tasklist, var_value2label, var_behlist,
                    subtract_psth="task", aligned_tbins=[26], neuron_idx=idx_top[i])
    plt.savefig(os.path.join(log_dir, f"neuron_{idx_top[i]}.png"))
    # wandb
    if config.wandb.use:
        wandb.log({f"neuron_{idx_top[i]}": wandb.Image(os.path.join(log_dir, f"neuron_{idx_top[i]}.png"))})
    