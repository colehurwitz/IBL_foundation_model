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

from behavior_decoders.decoder_loader import SingleSessionDataModule
from behavior_decoders.models import ReducedRankDecoder, MLPDecoder, LSTMDecoder
from behavior_decoders.eval import eval_model
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
    "model": "include:src/configs/behavior_decoder/decoder.yaml"
}

config = config_from_kwargs(kwargs)
config = update_config("src/configs/behavior_decoder/decoder.yaml", config)
config = update_config("src/configs/behavior_decoder/decoder_trainer.yaml", config)

# Need user inputs: choice of dataset & behavior
ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str)
ap.add_argument("--target", type=str)
ap.add_argument("--method", type=str)
ap.add_argument("--n_workers", type=int, default=1)
args = ap.parse_args()

# wandb
if config.wandb.use:
    import wandb
    wandb.login()
    wandb.init(
        # project=args.target, entity=args.eid, 
        config=config,
        name="train_{}".format(args.method)
    )

set_seed(config.seed)

save_path = Path(config.dirs.output_dir) / args.target / args.method 
os.makedirs(save_path, exist_ok=True)

"""
--------
DECODING
--------
"""

model_class = args.method

print(f'Decode {args.target} from session {args.eid}:')
print(f'Launch {model_class} decoder:')
print('----------------------------------------------------')

search_space = config.copy()
search_space['eid'] = args.eid
search_space['target'] = args.target
search_space['training']['device'] = torch.device(
    'cuda' if np.logical_and(torch.cuda.is_available(), config.training.device == 'gpu') else 'cpu'
)

if model_class == "linear":
    dm = SingleSessionDataModule(search_space)
    dm.setup()
    if config.model.target == 'reg':
        model = GridSearchCV(Ridge(), {"alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]})
    elif config.model.target == 'clf':
        model = GridSearchCV(LogisticRegression(), {"C": [1, 1e1, 1e2, 1e3, 1e4]})
    else:
        raise NotImplementedError
    metric, test_pred, test_y = eval_model(
        dm.train, dm.test, model, target=config.model.target, model_class=model_class
    )
else:
    def train_func(config):
        dm = SingleSessionDataModule(config)
        dm.setup()
        if model_class == "reduced-rank":
            model = ReducedRankDecoder(dm.config)
        elif model_class == "lstm":
            model = LSTMDecoder(dm.config)
        elif model_class == "mlp":
            model = MLPDecoder(dm.config)
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
    search_space['optimizer']['weight_decay'] = tune.grid_search([1, 1e-1, 1e-2, 1e-3])
    
    if model_class == "reduced-rank":
        search_space['reduced_rank']['temporal_rank'] = tune.grid_search([2, 5, 10, 15])
        search_space['tuner']['num_epochs'] = 500
        search_space['training']['num_epochs'] = 800
    elif model_class == "lstm":
        search_space['lstm']['lstm_hidden_size'] = tune.grid_search([128, 64])
        search_space['lstm']['lstm_n_layers'] = tune.grid_search([1, 3, 5])
        search_space['lstm']['drop_out'] = tune.grid_search([0., 0.2, 0.4, 0.6])
        search_space['tuner']['num_epochs'] = 250
        search_space['training']['num_epochs'] = 250
    elif model_class == "mlp":
        search_space['mlp']['drop_out'] = tune.grid_search([0., 0.2, 0.4, 0.6])
        search_space['tuner']['num_epochs'] = 250
        search_space['training']['num_epochs'] = 250
    else:
        raise NotImplementedError
    
    results = tune_decoder(
        train_func, search_space, use_gpu=config.tuner.use_gpu, max_epochs=config.tuner.num_epochs, 
        num_samples=config.tuner.num_samples, num_workers=args.n_workers
    )
    
    best_result = results.get_best_result(metric=config.tuner.metric, mode=config.tuner.mode)
    best_config = best_result.config['train_loop_config']

    print(best_config)
    
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
    dm = SingleSessionDataModule(best_config)
    dm.setup()
    
    if model_class == "reduced-rank":
        model = ReducedRankDecoder(best_config)
    elif model_class == "lstm":
        model = LSTMDecoder(best_config)
    elif model_class == "mlp":
        model = MLPDecoder(best_config)
    else:
        raise NotImplementedError
    
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm, ckpt_path='best')
    metrics = trainer.test(datamodule=dm, ckpt_path='best')[0]
    metric = metrics['test_metric']
    
    _, test_pred, test_y = eval_model(
        dm.train, dm.test, model, target=best_config['model']['target'], model_class=model_class
    )
    
print(f'{model_class} {args.target} test metric: ', metric)

if config.wandb.use:
    wandb.log(
        {"test_metric": metric, "test_pred": test_pred, "test_y": test_y}
    )
    wandb.finish()
else:
    res_dict = {'test_metric': metric, 'test_pred': test_pred, 'test_y': test_y}
    np.save(save_path / f'{args.eid}.npy', res_dict)
        
