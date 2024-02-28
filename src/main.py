from datasets import load_dataset
import numpy as np
from loader.make_loader import make_batcher

# download dataset from huggingface
train_dataset = load_dataset("berkott/ibl_ssl_data", cache_dir='../checkpoints/datasets_cache')

# show the columns
print(train_dataset.column_names)

# select the train dataset and the spikes_sparse_data column
train_dataset = train_dataset["train"].select_columns(['spikes_sparse_data'])

# make the dataloader
dataloader = make_batcher(train_dataset, batch_size=32, pad_to_ritght=True, max_pad=6000, shuffle=True)

# TODO: Add huggingface accelerator

# loop through the dataloader
for batch in dataloader:
    print(batch.keys())
    print(batch['spikes_sparse_data'].shape)
    print(batch['spikes_sparse_data'][-20:])
    break