from datasets import load_dataset, load_from_disk, concatenate_datasets
from utils.dataset_utils import load_ibl_dataset
from accelerate import Accelerator
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from loader.make_loader import make_loader
from utils.utils import (set_seed,move_batch_to_device, 
                         viz_single_cell, 
                         prep_cond_matrix,
                         var_name2idx,
                         var_value2label,
                         var_tasklist)
from utils.config_utils import config_from_kwargs, update_config
from models.ndt1 import NDT1
from models.stpatch import STPatch
from models.itransformer import iTransformer


# load config
kwargs = {
    "model": "include:src/configs/ndt2.yaml"
}

config = config_from_kwargs(kwargs)
config = update_config("src/configs/ndt2.yaml", config)
config = update_config("src/configs/ssl_session_trainer.yaml", config)

mask_mode = config.encoder.masker.mode

# make log dir
log_dir = os.path.join(config.dirs.log_dir, 
                       "eval", "model_{}".format(config.model.model_class), 
                       "method_{}".format(config.method.model_kwargs.method_name), 
                       "mask_{}".format(mask_mode))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# wandb
if config.wandb.use:
    import wandb
    wandb.init(project=config.wandb.project, entity=config.wandb.entity, config=config, name="eval_model_{}_method_{}_mask_{}".format(config.model.model_class, config.method.model_kwargs.method_name,mask_mode))

# set seed for reproducibility
set_seed(config.seed)

eid = "671c7ea7-6726-4fbe-adeb-f89c2c8e489b"
# download dataset from huggingface
_, _, test_dataset = load_ibl_dataset(config.dirs.dataset_cache_dir, 
                           config.dirs.huggingface_org,
                           aligned_data_dir=config.dirs.aligned_data_dir,
                           eid=eid,
                           num_sessions=config.data.num_sessions,
                           split_method=config.data.split_method,
                           test_session_eid=config.data.test_session_eid,
                           seed=config.seed)

# make the dataloader
test_dataloader = make_loader(test_dataset, 
                         batch_size=config.training.test_batch_size, 
                         pad_to_right=True, 
                         pad_value=-1.,
                         max_time_length=config.data.max_time_length,
                         max_space_length=config.data.max_space_length,
                         dataset_name=config.data.dataset_name,
                         shuffle=False)

# Initialize the accelerator
accelerator = Accelerator()

# load model
NAME2MODEL = {"NDT1": NDT1, "STPatch": STPatch, "iTransformer": iTransformer}
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
        if not config.data.patching:
            gt.append(batch['spikes_data'].clone())
            outputs = model(
                batch['spikes_data'], 
                batch['attention_mask'], 
                batch['spikes_timestamps']
            )
        else:
            outputs = model(
            batch['spikes_data'], 
            batch['time_attn_mask'],
            batch['space_attn_mask'],
            batch['spikes_timestamps'], 
            batch['spikes_spacestamps'], 
            targets = batch['target']
        )
            gt.append(outputs.targets.clone())
        preds.append(outputs.preds.clone())

gt = torch.cat(gt, dim=0)
preds = torch.cat(preds, dim=0)

if config.method.model_kwargs.loss == "poisson_nll":
    preds = torch.exp(preds)

behavior_set = prep_cond_matrix(test_dataset)

# Settings for validation
X = behavior_set # [#trials, #timesteps, #variables]
ys = gt.cpu().numpy() # [#trials, #timesteps, #neurons]
y_preds = preds.cpu().numpy() # [#trials, #timesteps, #neurons]

# choose more active neuron
tmp = np.mean(ys, axis=(0,1))
idx_top = np.argsort(tmp)[-5:]
if mask_mode == "co-smooth":
    idx_top = config.encoder.masker.channels
    idx_top = np.array(idx_top)

var_behlist = []

psth_r2_list = []
pred_r2_list = []
for i in range(idx_top.shape[0]):
    metrics = viz_single_cell(X,ys[:,:,idx_top[i]],y_preds[:,:,idx_top[i]], 
                    var_name2idx, var_tasklist, var_value2label, var_behlist,
                    subtract_psth="task", aligned_tbins=[26], neuron_idx=idx_top[i])
    
    psth_r2_list.append(metrics['psth_r2'])
    pred_r2_list.append(metrics['pred_r2'])
    plt.savefig(os.path.join(log_dir, f"neuron_{idx_top[i]}.png"))
    # wandb
    if config.wandb.use:
        wandb.log({f"neuron_{idx_top[i]}": wandb.Image(os.path.join(log_dir, f"neuron_{idx_top[i]}.png"))})

print("Mean PSTH R2: {}, STD PSTH R2: {}".format(np.mean(psth_r2_list), np.std(psth_r2_list)))
print("Mean Pred R2: {}, STD Pred R2: {}".format(np.mean(pred_r2_list), np.std(pred_r2_list)))
if config.wandb.use:
    wandb.log({"mean_psth_r2": np.mean(psth_r2_list), 
               "std_psth_r2": np.std(psth_r2_list),
               "mean_pred_r2": np.mean(pred_r2_list),
                "std_pred_r2": np.std(pred_r2_list)})
    