import os
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

def tune_decoder(
    train_func, search_space,
    max_epochs=100, num_samples=1, use_gpu=False, num_workers=1, 
    metric="loss", mode="min", 
):
    
    if use_gpu:
        resources_per_worker={"CPU": 1, "GPU": 1}
    else:
        resources_per_worker={"CPU": 1}

    scaling_config = ScalingConfig(
        num_workers=num_workers, use_gpu=use_gpu, resources_per_worker=resources_per_worker
    )
    
    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute=metric,
            checkpoint_score_order=mode,
        ),
    )
    
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    
    scheduler = ASHAScheduler(max_t=max_epochs, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric=metric,
            mode=mode,
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()
    