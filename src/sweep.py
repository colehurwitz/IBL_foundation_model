import wandb
from utils.config_utils import config_from_kwargs, update_config
from train_lib import train

if __name__ == '__main__':

    kwargs = {"model": "include:src/configs/sweep.yaml"}
    
    sweep_config = config_from_kwargs(kwargs)
    sweep_config = update_config("src/configs/sweep.yaml", sweep_config)
    print(sweep_config)
    
    sweep_id = wandb.sweep(sweep_config.copy(), project=sweep_config.project)
    wandb.agent(sweep_id, function=train)

    api = wandb.Api()
    sweep = api.sweep(f"yz4123/{sweep_config.project}/sweeps/{sweep_id}") # Replace with your own wandb setting
    
    best_run = sweep.best_run(order='eval_loss')
    print(best_run.config)

    train(sweep=False, **best_run.config)
    
    