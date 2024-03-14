from datasets import load_dataset
from accelerate import Accelerator
from loader.make_loader import make_loader
from utils.utils import set_seed, move_batch_to_device, plot_gt_pred, metrics_list, plot_avg_rate_and_spike, plot_rate_and_spike, plt_condition_avg_r2
from utils.utils import set_seed, move_batch_to_device, plot_gt_pred, metrics_list, plot_r2
from utils.config_utils import config_from_kwargs, update_config
from utils.dataset_utils import get_data_from_h5
from models.ndt1 import NDT1
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import torch
import numpy as np
import copy
import os

# load config
kwargs = {
    "model": "include:src/configs/ndt1.yaml"
}

config = config_from_kwargs(kwargs)
config = update_config("src/configs/ndt1.yaml", config)
config = update_config("src/configs/trainer.yaml", config)

# make log dir
log_dir = os.path.join(config.dirs.log_dir, "model_{}".format(config.model.model_class), "method_{}".format(config.method.model_kwargs.method_name))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# wandb
if config.wandb.use:
    import wandb
    wandb.init(project=config.wandb.project, entity=config.wandb.entity, config=config, name=config.wandb.run_name)

# set seed for reproducibility
set_seed(config.seed)

# download dataset from huggingface
if "ibl" in config.data.dataset_name:
    dataset = load_dataset(config.dirs.dataset_dir, cache_dir=config.dirs.dataset_cache_dir)
    # show the columns
    print(dataset.column_names)
    bin_size = dataset["train"]["bin_size"][0]
    print(f"bin_size: {bin_size}")

    # split the dataset to train and test
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=config.seed)
    # select the train dataset and the spikes_sparse_data column
    train_dataset = dataset["train"].select_columns(['spikes_sparse_data', 'spikes_sparse_indices', 'spikes_sparse_indptr', 'spikes_sparse_shape'])
    test_dataset = dataset["test"].select_columns(['spikes_sparse_data', 'spikes_sparse_indices', 'spikes_sparse_indptr', 'spikes_sparse_shape'])
else:
    train_dataset = get_data_from_h5("train", config.dirs.dataset_dir, config=config)
    test_dataset = get_data_from_h5("val", config.dirs.dataset_dir, config=config)
    bin_size = None

# sample a neuron index
sampled_neuron_idx = np.random.randint(0, config.encoder.embedder.n_channels)
sampled_neuron_idx = 0
print(f"sampled_neuron_idx: {sampled_neuron_idx}")

# make the dataloader
train_dataloader = make_loader(train_dataset, 
                         batch_size=config.training.train_batch_size, 
                         pad_to_right=True, 
                         pad_value=-1.,
                         bin_size=bin_size,
                         max_length=config.data.max_seq_len,
                         dataset_name=config.data.dataset_name,
                         shuffle=True)

test_dataloader = make_loader(test_dataset, 
                         batch_size=config.training.test_batch_size, 
                         pad_to_right=True, 
                         pad_value=-1.,
                         bin_size=bin_size,
                         max_length=config.data.max_seq_len,
                         dataset_name=config.data.dataset_name,
                         shuffle=False)

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
                total_steps=config.training.num_epochs*len(train_dataloader) * 100 //config.optimizer.gradient_accumulation_steps,
                max_lr=config.optimizer.lr,
                pct_start=config.optimizer.warmup_pct,
                div_factor=config.optimizer.div_factor,
            )
best_test_trial_avg_r2 = -np.inf
# loop through the dataloader
for epoch in range(config.training.num_epochs):
    train_loss = 0.
    train_examples = 0
    model.train()
    for batch in train_dataloader:
        batch = move_batch_to_device(batch, accelerator.device)
        outputs = model(batch['spikes_data'], batch['attention_mask'], batch['spikes_timestamps'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        train_loss += loss.item()
        train_examples += outputs.n_examples
    
    print(f"epoch: {epoch} train loss: {train_loss/train_examples}")

    test_loss = 0.
    test_examples = 0
    model.eval()
    test_pred_list = []
    with torch.no_grad():
        for batch in test_dataloader:
            batch = move_batch_to_device(batch, accelerator.device)
            spike_data_copy = copy.deepcopy(batch['spikes_data'])
            outputs = model(spike_data_copy, batch['attention_mask'], batch['spikes_timestamps'])
            outputs = model(batch['spikes_data'], batch['attention_mask'], batch['spikes_timestamps'])
            loss = outputs.loss
            test_loss += loss.item()
            test_examples += outputs.n_examples
            test_pred_list.append((torch.exp(outputs.preds), batch['spikes_data']))

    if config.data.use_lograte:
        preds = torch.exp(outputs.preds)
        if "rates" in batch:
            gt = torch.exp(batch['rates'])
        else:
            gt = batch['spikes_data']
    else:
        preds = outputs.preds
        gt = batch['spikes_data']

    # calculate results metrics
    results = metrics_list(gt = gt[..., sampled_neuron_idx],
                           pred = preds[..., sampled_neuron_idx],
                           metrics=["r2"],
                           device=accelerator.device)
    print(f"epoch: {epoch} test_loss: {test_loss/test_examples} r2: {results['r2']}")

    if config.wandb.use:
        logs = {
            "epoch": epoch, 
            "train_loss": train_loss/train_examples, 
            "test_loss": test_loss/test_examples,
            "lr": optimizer.param_groups[0]['lr'],
        }
        # merge the results with the logs
        logs.update(results)
        wandb.log(logs)
    
    if epoch % config.training.save_plot_every_n_epochs == 0:
        # plot the r2 score of a sampled neuron in single trial
        fig_r2 = plot_r2(gt = gt[0, :, sampled_neuron_idx],
                    pred = preds[0, :, sampled_neuron_idx],
                    neuron_idx=sampled_neuron_idx,
                    epoch = epoch,
                    )
        
        # plot Ground Truth and Prediction of a single trial
        fig_gt_pred = plot_gt_pred(gt = gt[0].T.cpu().numpy(),
                    pred = preds[0].T.detach().cpu().numpy(),
                    epoch = epoch)
        
        # plot condition average r2 score
        fig_condition_avg_r2 = plt_condition_avg_r2(gt = gt,
                    pred = preds,
                    neuron_idx=sampled_neuron_idx,
                    condition_idx=0,
                    first_n=8,
                    device=accelerator.device,
                    epoch = epoch)
        
        if config.wandb.use:
            wandb.log({"epoch": epoch,
                       "test_gt_pred_single_trial": [wandb.Image(fig_gt_pred)],
                       "test_r2_score_single_trial": [wandb.Image(fig_r2)],
                       "test_condition_avg_r2": [wandb.Image(fig_condition_avg_r2)]})
        else:
            fig_r2.savefig(os.path.join(log_dir, f"test_r2_score_single_trial_epoch_{epoch}.png"))
            fig_gt_pred.savefig(os.path.join(log_dir, f"test_gt_pred_single_trial_epoch_{epoch}.png"))
            fig_condition_avg_r2.savefig(os.path.join(log_dir, f"test_condition_avg_r2_epoch_{epoch}.png"))

    if results['r2'] > best_test_trial_avg_r2:
        best_test_trial_avg_r2 = results['r2']
        print(f"Epoch {epoch} Save the best model with r2: {best_test_trial_avg_r2}")
        # save the model
        torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))

if config.wandb.use:
    wandb.log({"best_test_trial_avg_r2": best_test_trial_avg_r2})
# save the model
torch.save(model.state_dict(), os.path.join(log_dir, "last_model.pth"))


    # if "rates" not in batch:
    #     # plot the GT rate and inferred rate
    #     fig_rate = plot_rate_and_spike(
    #         gt=batch['spikes_data'][1].T.cpu().numpy(),
    #         pred=outputs.preds[1].T.detach().cpu().numpy(),
    #         epoch=epoch
    #     )

    #     # plot the avg inferred rate and spike count
    #     fig_avg = plot_avg_rate_and_spike(test_pred_list, epoch)



    