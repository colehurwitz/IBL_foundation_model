import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer

from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)

from decoder_loader import SingleSessionDataModule
from models import ReducedRankDecoder, MLPDecoder, LSTMDecoder
from eval import eval_model
from hyperparam_tuning import tune_decoder

from ray import tune

from utils.utils import set_seed
from utils.config_utils import config_from_kwargs, update_config

"""
-------
CONFIGS
-------
"""

kwargs = {
    "model": "include:src/configs/decoder.yaml"
}

config = config_from_kwargs(kwargs)
config = update_config("src/configs/decoder.yaml", config)
config = update_config("src/configs/decoder_trainer.yaml", config)

set_seed(config.seed)

# Need user inputs: choice of dataset & behavior
ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str)
ap.add_argument("--target", type=str)
args = ap.parse_args()

config.eid = args.eid
config.target = args.target

# Load cached Hugging Face dataset
# 

"""
--------
DECODING
--------
"""

print(f'Decode {args.target} from session {args.eid}:')
print('----------------------------------------------------')

def save_results(model_class, r2, test_pred, test_y):
    res_dict = {'r2': r2, 'pred': test_pred, 'target': test_y}
    save_path = Path(config.dirs.output_dir) / args.target / model_class 
    os.makedirs(save_path, exist_ok=True)
    np.save(save_path / f'{args.eid}.npy', res_dict)
    print(f'{model_class} {args.target} test R2: ', r2)

for model_class in ['ridge', 'reduced_rank', 'lstm', 'mlp']:

    print(f'Launch {model_class} decoder:')
    print('----------------------------------------------------')

    if model_type == "ridge":
        dm = SingleSessionDataModule(config)
        dm.setup()
        alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
        model = GridSearchCV(Ridge(), {"alpha": alphas})
        r2, test_pred, test_y = eval_model(dm.train, dm.test, model, model_class=model_class, plot=False)
        save_results(model_class, r2, test_pred, test_y)
        continue

    def train_func(config):
        dm = SingleSessionDataModule(config)
        dm.setup()
        if model_class == "reduced_rank":
            model = ReducedRankDecoder(dm.config)
        elif model_class == "lstm":
            model = LSTMDecoder(dm.config)
        elif model_class == "mlp":
            model = MLPDecoder(dm.config)
        else:
            raise NotImplementedError
    
        trainer = Trainer(
            max_epochs=config.tuner.num_epochs,
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
            enable_progress_bar=config.tuner.enable_progress_bar,
        )
        trainer = prepare_trainer(trainer)
        trainer.fit(model, datamodule=dm)

    if model_type == "reduced_rank":
        search_space = config.copy()
        search_space.reduced_rank.temporal_rank = tune.grid_search([2, 5, 10])
    elif model_type == "lstm":
        search_space = config.copy()
        search_space.lstm.lstm_hidden_size = tune.grid_search([32, 64])
        search_space.lstm.lstm_n_layers = tune.grid_search([1, 3, 5])
        search_space.lstm.mlp_hidden_size = tune.grid_search([(32,), (64,)])
    elif model_type == "mlp":
        search_space = config.copy()
        search_space.mlp.mlp_hidden_size = tune.grid_search([(256, 128, 64), (512, 256, 128, 64)])
    else:
        raise NotImplementedError

    results = tune_decoder(
        train_func, search_space, use_gpu=config.tuner.use_gpu, max_epochs=config.tuner.num_epochs, 
        num_samples=config.tuner.num_epochs, num_workers=config.tuner.num_workers
    )
    
    best_result = results.get_best_result(metric=config.tuner.metric, mode=config.tuner.mode)
    best_config = best_result.config['train_loop_config']

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
    
    if model_type == "reduced_rank":
        model = ReducedRankDecoder(best_config)
    elif model_type == "lstm":
        model = LSTMDecoder(best_config)
    elif model_type == "mlp":
        model = MLPDecoder(best_config)
    else:
        raise NotImplementedError
    
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm, ckpt_path='best')

    r2, test_pred, test_y = eval_model(dm.train, dm.test, model, model_class)
    save_results(model_class, r2, test_pred, test_y)

