import torch
import numpy as np
import wandb
import os
from utils.utils import move_batch_to_device, metrics_list, plot_gt_pred, plot_neurons_r2
from tqdm import tqdm
import random

class Trainer():
    def __init__(
            self,
            model,
            train_dataloader,
            eval_dataloader,
            optimizer,
            **kwargs
    ):
        # get all the arguments
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer

        # get arguments from kwargs if they exist
        self.log_dir = kwargs.get("log_dir", None)
        self.accelerator = kwargs.get("accelerator", None)
        self.lr_scheduler = kwargs.get("lr_scheduler", None)
        self.config = kwargs.get("config", None)

        if self.config.method.model_kwargs.clf:
            self.metric = 'acc'
        else:
            self.metric = 'r2'
                
        self.active_neurons = None

        self.masking_ratio = model.encoder.masker.ratio
        self.masking_mode = model.encoder.masker.mode
        self.masking_schemes = ['neuron', 'temporal', 'causal']
        if self.masking_mode == "all":
            self.masking_schemes += ['intra-region', 'inter-region']
        if self.config.model.model_class == 'STPatch':
            self.masking_schemes += ['random_token']

        if self.masking_mode in ["combined", "all"]:
            print("(train) switch between masking modes: ", self.masking_schemes)

    def train(self):
        best_eval_loss = torch.tensor(float('inf'))
        best_eval_trial_avg_metric = -torch.tensor(float('inf'))
        # train loop
        for epoch in range(self.config.training.num_epochs):
            train_epoch_results = self.train_epoch(epoch)
            eval_epoch_results = self.eval_epoch()
            print(f"epoch: {epoch} train loss: {train_epoch_results['train_loss'] }")

            if eval_epoch_results:
                if eval_epoch_results[f'eval_trial_avg_{self.metric}'] > best_eval_trial_avg_metric:
                    best_eval_trial_avg_metric = eval_epoch_results[f'eval_trial_avg_{self.metric}']
                    print(f"epoch: {epoch} best eval trial avg {self.metric}: {best_eval_trial_avg_metric}")
                    # save model
                    self.save_model(name="best", epoch=epoch)
                    if self.config.method.model_kwargs.method_name == 'ssl':
                        gt_pred_fig = self.plot_epoch(
                            gt=eval_epoch_results['eval_gt'], 
                            preds=eval_epoch_results['eval_preds'], epoch=epoch
                        )
                        if self.config.wandb.use:
                            wandb.log({"best_epoch": epoch,
                                    "best_gt_pred_fig": wandb.Image(gt_pred_fig['plot_gt_pred']),
                                    "best_r2_fig": wandb.Image(gt_pred_fig['plot_r2'])})
                        else:
                            gt_pred_fig['plot_gt_pred'].savefig(
                                os.path.join(self.log_dir, f"best_gt_pred_fig_{epoch}.png")
                            )
                            gt_pred_fig['plot_r2'].savefig(
                                os.path.join(self.log_dir, f"best_r2_fig_{epoch}.png")
                            )
                if eval_epoch_results['eval_loss'] < best_eval_loss:
                    best_eval_loss = eval_epoch_results['eval_loss']
                    print(f"epoch: {epoch} best eval loss: {best_eval_loss}")
                print(f"epoch: {epoch} eval loss: {eval_epoch_results['eval_loss']} {self.metric}: {eval_epoch_results[f'eval_trial_avg_{self.metric}']}")

            # save model by epoch
            if epoch % self.config.training.save_every == 0:
                self.save_model(name="epoch", epoch=epoch)

            # plot epoch
            if epoch % self.config.training.save_plot_every_n_epochs == 0:
                if self.config.method.model_kwargs.method_name == 'ssl':
                    gt_pred_fig = self.plot_epoch(
                        gt=eval_epoch_results['eval_gt'], 
                        preds=eval_epoch_results['eval_preds'], 
                        epoch=epoch
                    )
                    if self.config.wandb.use:
                        wandb.log({
                            "gt_pred_fig": wandb.Image(gt_pred_fig['plot_gt_pred']),
                            "r2_fig": wandb.Image(gt_pred_fig['plot_r2'])
                        })
                    else:
                        gt_pred_fig['plot_gt_pred'].savefig(
                            os.path.join(self.log_dir, f"gt_pred_fig_{epoch}.png")
                        )
                        gt_pred_fig['plot_r2'].savefig(
                            os.path.join(self.log_dir, f"r2_fig_{epoch}.png")
                        )

            # wandb log
            if self.config.wandb.use:
                wandb.log({
                    "train_loss": train_epoch_results['train_loss'],
                    "eval_loss": eval_epoch_results['eval_loss'],
                    f"eval_trial_avg_{self.metric}": eval_epoch_results[f'eval_trial_avg_{self.metric}']
                })
                
        # save last model
        self.save_model(name="last", epoch=epoch)
        
        if self.config.wandb.use:
            wandb.log({"best_eval_loss": best_eval_loss,
                       f"best_eval_trial_avg_{self.metric}": best_eval_trial_avg_metric})
            
    def train_epoch(self, epoch):
        train_loss = 0.
        train_examples = 0
        self.model.train()
        for batch in tqdm(self.train_dataloader):
            if self.masking_mode in ["combined", "all"]:
                masking_mode = random.sample(self.masking_schemes, 1)[0]
                if masking_mode == 'temporal':
                    self.model.encoder.masker.ratio = 0.3
                elif masking_mode == 'causal':
                    self.model.encoder.masker.ratio = 1.0
                else:
                    self.model.encoder.masker.ratio = self.masking_ratio
            else:
                masking_mode = self.masking_mode
            outputs = self._forward_model_outputs(batch, masking_mode)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            train_loss += loss.item()
            train_examples += outputs.n_examples
        return{
            "train_loss": train_loss/train_examples
        }
    
    def _forward_model_outputs(self, batch, masking_mode):
        batch = move_batch_to_device(batch, self.accelerator.device)
        return self.model(
            batch['spikes_data'], 
            time_attn_mask=batch['time_attn_mask'],
            space_attn_mask=batch['space_attn_mask'],
            spikes_timestamps=batch['spikes_timestamps'], 
            spikes_spacestamps=batch['spikes_spacestamps'], 
            targets = batch['target'],
            neuron_regions=batch['neuron_regions'],
            masking_mode=masking_mode, 
            spike_augmentation=self.config.data.spike_augmentation
        ) 
    
    def eval_epoch(self):
        self.model.eval()
        eval_loss = 0.
        eval_examples = 0
        if self.eval_dataloader:
            gt, preds = [], []
            with torch.no_grad():  
                for batch in self.eval_dataloader:
                    if self.masking_mode in ["combined", "all"]:
                        masking_mode = random.sample(self.masking_schemes, 1)[0]
                        if masking_mode == 'temporal':
                            self.model.encoder.masker.ratio = 0.3
                        elif masking_mode == 'causal':
                            self.model.encoder.masker.ratio = 1.0
                        else:
                            self.model.encoder.masker.ratio = self.masking_ratio
                    else:
                        masking_mode = self.masking_mode
                    outputs = self._forward_model_outputs(batch, masking_mode)
                    loss = outputs.loss
                    eval_loss += loss.item()
                    eval_examples += outputs.n_examples
                    gt.append(outputs.targets.clone())
                    preds.append(outputs.preds.clone())
            gt = torch.cat(gt, dim=0)
            preds = torch.cat(preds, dim=0)
            
        if self.config.method.model_kwargs.loss == "poisson_nll":
            preds = torch.exp(preds)
        elif self.config.method.model_kwargs.loss == "cross_entropy":
            preds = torch.nn.functional.softmax(preds, dim=1)
            
        if self.active_neurons is None:
            self.active_neurons = np.argsort(gt.cpu().numpy().sum((0,1)))[::-1][:5].tolist()

        if self.config.method.model_kwargs.method_name == 'ssl':
            results = metrics_list(gt = gt.mean(0)[..., self.active_neurons].T,
                                   pred = preds.mean(0)[..., self.active_neurons].T, 
                                   metrics=["r2"], 
                                   device=self.accelerator.device)
        elif self.config.method.model_kwargs.method_name == 'sl':
            if self.config.method.model_kwargs.clf:
                results = metrics_list(gt = gt.argmax(1),
                                       pred = preds.argmax(1), 
                                       metrics=[self.metric], 
                                       device=self.accelerator.device)
            elif self.config.method.model_kwargs.reg:
                results = metrics_list(gt = gt,
                                       pred = preds,
                                       metrics=[self.metric],
                                       device=self.accelerator.device)

        return {
            "eval_loss": eval_loss/eval_examples,
            f"eval_trial_avg_{self.metric}": results[self.metric],
            "eval_gt": gt,
            "eval_preds": preds,
        }
    
    def plot_epoch(self, gt, preds, epoch):
        gt_pred_fig = plot_gt_pred(gt = gt.mean(0).T.cpu().numpy(),
                     pred = preds.mean(0).T.detach().cpu().numpy(),
                     epoch = epoch)
        
        r2_fig = plot_neurons_r2(gt = gt.mean(0),
                pred = preds.mean(0),
                neuron_idx=self.active_neurons,
                epoch = epoch)
        return {
            "plot_gt_pred": gt_pred_fig,
            "plot_r2": r2_fig
        }
        

    def save_model(self, name="last", epoch=0):
        # save model
        print(f"saving model: {name} to {self.log_dir}")
        dict_config = {
            "model": self.model,
            "epoch": epoch,
        }
        torch.save(dict_config, os.path.join(self.log_dir, f"model_{name}.pt"))
