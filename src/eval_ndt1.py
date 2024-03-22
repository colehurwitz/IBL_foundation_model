from datasets import load_dataset
from accelerate import Accelerator
from loader.make_loader import make_loader
from utils.utils import set_seed, move_batch_to_device, plot_gt_pred, metrics_list, plot_avg_rate_and_spike, plot_rate_and_spike, plt_condition_avg_r2
from utils.utils import set_seed, move_batch_to_device, plot_gt_pred, plot_neurons_r2
from utils.config_utils import config_from_kwargs, update_config
from utils.dataset_utils import get_data_from_h5
from models.ndt1 import NDT1
import matplotlib.pyplot as plt
import torch
import numpy as np
import os

# load config
kwargs = {
    "model": "include:src/configs/ndt1.yaml"
}

config = config_from_kwargs(kwargs)
config = update_config("src/configs/ndt1.yaml", config)
config = update_config("src/configs/trainer.yaml", config)

# make log dir
log_dir = os.path.join(config.dirs.log_dir, "model_{}".format(config.model.model_class), "eval_{}".format(config.data.dataset_name))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# wandb
if config.wandb.use:
    import wandb
    wandb.init(project=config.wandb.project, entity=config.wandb.entity, config=config, name=config.wandb.run_name)

# set seed for reproducibility
set_seed(config.seed)

train_dataset = get_data_from_h5("train", config.dirs.dataset_dir, config=config)
test_dataset = get_data_from_h5("val", config.dirs.dataset_dir, config=config)

# make the dataloader
train_dataloader = make_loader(train_dataset, 
                         batch_size=2048, 
                         pad_to_right=True, 
                         pad_value=-1.,
                         max_time_length=config.data.max_seq_len,
                         dataset_name=config.data.dataset_name,
                         shuffle=False)

# Initialize the accelerator
accelerator = Accelerator()

# load model
NAME2MODEL = {"NDT1": NDT1}
model_class = NAME2MODEL[config.model.model_class]
model = model_class(config.model, **config.method.model_kwargs)
# load pretrained model

model.load_state_dict(torch.load(config.dirs.pretrained_model_path))
model = accelerator.prepare(model)

with torch.no_grad():
    model.eval()
    for batch in train_dataloader:
        batch = move_batch_to_device(batch, accelerator.device)
        outputs = model(batch['spikes_data'], batch['attention_mask'], batch['spikes_timestamps'])
        loss = outputs.loss/outputs.n_examples

        print(f"train_loss: {loss.item()}")
        if config.wandb.use:
            wandb.log({"train_loss": loss.item()})

        if config.data.use_lograte:
            preds = torch.exp(outputs.preds)
            if "rates" in batch:
                gt = torch.exp(batch['rates'])
            else:
                gt = batch['spikes_data']
        else:
            preds = outputs.preds
            gt = batch['spikes_data']
        
        for neuron in range(gt.shape[2]):
            # plot
            # plot the r2 score of a sampled neuron in single trial
            fig_r2 = plot_neurons_r2(gt = gt[0, :, neuron],
                        pred = preds[0, :, neuron],
                        neuron_idx=neuron,
                        epoch = 0,
                        )
            
            # plot Ground Truth and Prediction of a single trial
            # trial_idx = 0
            fig_gt_pred = plot_gt_pred(gt = gt[0].T.cpu().numpy(),
                        pred = preds[0].T.detach().cpu().numpy(),
                        epoch = 0)
            
            # plot condition average r2 score
            fig_condition_avg_r2 = plt_condition_avg_r2(gt = gt,
                        pred = preds,
                        neuron_idx=neuron,
                        condition_idx=1,
                        first_n=8,
                        device=accelerator.device,
                        epoch = 0)
            
            if config.wandb.use:
                wandb.log({"neuron": neuron,
                        "test_gt_pred_single_trial": [wandb.Image(fig_gt_pred)],
                        "test_r2_score_single_trial": [wandb.Image(fig_r2)],
                        "test_condition_avg_r2": [wandb.Image(fig_condition_avg_r2)]})
            else:
                fig_r2.savefig(os.path.join(log_dir, f"test_r2_score_single_trial_neuron_{neuron}.png"))
                fig_gt_pred.savefig(os.path.join(log_dir, f"test_gt_pred_single_trial_neuron_{neuron}.png"))
                fig_condition_avg_r2.savefig(os.path.join(log_dir, f"test_condition_avg_r2_neuron_{0}.png"))