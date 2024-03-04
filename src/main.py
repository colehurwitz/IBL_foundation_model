from datasets import load_dataset
from accelerate import Accelerator
from loader.make_loader import make_loader
from utils.utils import set_seed, move_batch_to_device
from utils.config_utils import config_from_kwargs, update_config
from models.ndt1 import NDT1
from torch.optim.lr_scheduler import OneCycleLR
import torch
import numpy as np

# load config
kwargs = {
    "model": "include:src/configs/ndt1.yaml"
}

config = config_from_kwargs(kwargs)
config = update_config("src/configs/ndt1.yaml", config)
config = update_config("src/configs/trainer.yaml", config)

# set seed for reproducibility
set_seed(config.seed)

# download dataset from huggingface
train_dataset = load_dataset(config.dirs.dataset_dir, cache_dir=config.dirs.dataset_cache_dir)

# show the columns
print(train_dataset.column_names)
bin_size = train_dataset["train"]["bin_size"][0]
print(f"bin_size: {bin_size}")

# select the train dataset and the spikes_sparse_data column
train_dataset = train_dataset["train"].select_columns(['spikes_sparse_data', 'spikes_sparse_indices', 'spikes_sparse_indptr', 'spikes_sparse_shape'])

# make the dataloader
dataloader = make_loader(train_dataset, 
                         batch_size=32, 
                         pad_to_right=True, 
                         pad_value=-1.,
                         bin_size=bin_size,
                         max_length=100, 
                         shuffle=True)


# Initialize the accelerator
accelerator = Accelerator()

# load model
NAME2MODEL = {"NDT1": NDT1}
model_class = NAME2MODEL[config.model.model_class]
model = model_class(config.model, **config.method.model_kwargs)
model = accelerator.prepare(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.wd, eps=config.optimizer.eps)
lr_scheduler = OneCycleLR(
                optimizer=optimizer,
                total_steps=config.training.num_epochs*len(dataloader) * 100 //config.optimizer.gradient_accumulation_steps,
                max_lr=config.optimizer.lr,
                pct_start=config.optimizer.warmup_pct,
                div_factor=config.optimizer.div_factor,
            )

# loop through the dataloader
for epoch in range(100):
    print(f"epoch: {epoch}")
    for batch in dataloader:
        batch = move_batch_to_device(batch, accelerator.device)
        # print(batch.keys())
        # print(batch['binned_spikes_data'].shape)
        # print(batch['attention_mask'].shape)
        outputs = model(batch['binned_spikes_data'], batch['attention_mask'], batch['spikes_timestamps'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print(f"loss: {loss.item()/outputs.n_examples}")

    