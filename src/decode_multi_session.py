import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, LogisticRegression

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer

from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)

from behavior_decoders.decoder_loader import MultiSessionDataModule
from behavior_decoders.models import MultiSessionReducedRankDecoder
from behavior_decoders.eval import eval_multi_session_model
from behavior_decoders.hyperparam_tuning import tune_decoder

from ray import tune

from utils.utils import set_seed
from utils.config_utils import config_from_kwargs, update_config


"""
-------
CONFIGS
-------
"""

kwargs = {
    "model": "include:configs/decoder.yaml"
}

config = config_from_kwargs(kwargs)
config = update_config("configs/decoder.yaml", config)
config = update_config("configs/decoder_trainer.yaml", config)

# Need user inputs: choice of dataset & behavior
ap = argparse.ArgumentParser()
ap.add_argument("--target", type=str)
ap.add_argument("--method", type=str)
ap.add_argument("--n_workers", type=int, default=1)
args = ap.parse_args()

# wandb
if config.wandb.use:
    import wandb
    wandb.login()
    wandb.init(
        project=args.target, entity='multi_sess', config=config,
        name="train_{}".format(args.method)
    )

set_seed(config.seed)

save_path = Path(config.dirs.output_dir) / args.target / ('multi-sess-' + args.method) 
os.makedirs(save_path, exist_ok=True)

"""
---------
LOAD DATA
---------
"""

eids = [fname.split('.')[0] for fname in os.listdir(config.dirs.data_dir)]

"""
--------
DECODING
--------
"""

model_class = args.method

print(f'Decode {args.target} from {len(eids)} sessions:')
print(f'Launch {model_class} decoder:')
print('----------------------------------------------------')

search_space = config.copy()
search_space['target'] = args.target
search_space['training']['device'] = torch.device(
    'cuda' if np.logical_and(torch.cuda.is_available(), config.training.device == 'gpu') else 'cpu'
)

def train_func(config):
    
    configs = []
    for eid in eids:
        _config = config.copy()
        _config['eid'] = eid
        configs.append(_config)
    
    dm = MultiSessionDataModule(eids, configs)
    dm.setup()
    
    base_config = dm.configs[0].copy()
    base_config['n_units'] = [_config['n_units'] for _config in dm.configs]

    if model_class == "reduced-rank":
        model = MultiSessionReducedRankDecoder(base_config)
    else:
        raise NotImplementedError

    trainer = Trainer(
        max_epochs=config['tuner']['num_epochs'],
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=config['tuner']['enable_progress_bar'],
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)

# -- Hyper parameter tuning 
# -------------------------

search_space['optimizer']['lr'] = tune.grid_search([1e-2, 1e-3])
search_space['optimizer']['weight_decay'] = tune.grid_search([0, 1e-1, 1e-2, 1e-3, 1e-4])

if model_class == "reduced-rank":
    search_space['temporal_rank'] = tune.grid_search([2, 5, 10, 15, 20])
    search_space['tuner']['num_epochs'] = 500
    search_space['training']['num_epochs'] = 800
else:
    raise NotImplementedError

results = tune_decoder(
    train_func, search_space, use_gpu=config.tuner.use_gpu, max_epochs=config.tuner.num_epochs, 
    num_samples=config.tuner.num_samples, num_workers=args.n_workers
)

best_result = results.get_best_result(metric=config.tuner.metric, mode=config.tuner.mode)
best_config = best_result.config['train_loop_config']

# -- Model training 
# -----------------
checkpoint_callback = ModelCheckpoint(
    monitor=config.training.metric, mode=config.training.mode, dirpath=config.dirs.checkpoint_dir
)

trainer = Trainer(
    max_epochs=config.training.num_epochs, 
    callbacks=[checkpoint_callback], 
    enable_progress_bar=config.training.enable_progress_bar
)

configs = []
for eid in eids:
    config = best_config.copy()
    config['eid'] = eid
    configs.append(config)

dm = MultiSessionDataModule(eids, configs)
dm.setup()

best_config = dm.configs[0].copy()
best_config['n_units'] = [_config['n_units'] for _config in dm.configs]
    
if model_class == "reduced-rank":
    model = MultiSessionReducedRankDecoder(best_config)
else:
    raise NotImplementedError

trainer.fit(model, datamodule=dm)
trainer.test(datamodule=dm, ckpt_path='best')

# -- Model Eval 
# -------------
metric_lst, test_pred_lst, test_y_lst = eval_multi_session_model(
    dm.train, dm.test, model, target=best_config['model']['target'], 
)

for eid_idx, eid in enumerate(eids):
    print(f'{eid} {args.target} test metric: ', metric)
    
if config.wandb.use:
    wandb.log(
        {"eids": eids, "test_metric": metric_lst, "test_pred": test_pred_lst, "test_y": test_y_lst}
    )
    wandb.finish()
else:
    for eid_idx, eid in enumerate(eids):
        res_dict = {
            'test_metric': metric_lst[eid_idx], 
            'test_pred': test_pred_lst[eid_idx], 
            'test_y': test_y_lst[eid_idx]
        }
        np.save(save_path / f'{eid}.npy', res_dict)
        
