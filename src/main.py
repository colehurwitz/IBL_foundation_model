from datasets import load_dataset
import numpy as np
from loader.make_loader import make_loader
from utils import set_seed

# set seed for reproducibility
set_seed(42)

# download dataset from huggingface
train_dataset = load_dataset("berkott/ibl_ssl_data", cache_dir='../checkpoints/datasets_cache')

# show the columns
print(train_dataset.column_names)

# select the train dataset and the spikes_sparse_data column
train_dataset = train_dataset["train"].select_columns(['spikes_sparse_data'])

# make the dataloader
dataloader = make_loader(train_dataset, 
                         batch_size=32, 
                         pad_to_right=True, 
                         pad_value=-1.,
                         max_length=6000, 
                         shuffle=True)

# TODO: Add huggingface accelerator

# loop through the dataloader
for batch in dataloader:
    print(batch.keys())
    print(batch['spikes_sparse_data'].shape)
    print(batch['spikes_sparse_data'][-5:,-20:])

    print(batch['attention_mask'].shape)
    print(batch['attention_mask'][-5:, -20:])
    break