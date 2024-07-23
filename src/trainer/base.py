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
        self.stitching = kwargs.get("stitching", None)
        self.num_neurons = kwargs.get("num_neurons", None)

        self.model_class = self.config.model.model_class

        if self.config.method.model_kwargs.clf:
            self.metric = 'acc'
        elif self.config.method.model_kwargs.reg:
            self.metric = 'rsquared'
        else:
            self.metric = 'r2'
                
        self.session_active_neurons = []

        self.masking_ratio = model.encoder.masker.ratio
        self.masking_mode = model.encoder.masker.mode
        self.masking_schemes = ['neuron', 'causal']
        if self.masking_mode == "all":
            self.masking_schemes += ['intra-region', 'inter-region']

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
                # if eval_epoch_results[f'eval_loss'] < best_eval_loss:
                    best_eval_loss = eval_epoch_results[f'eval_loss']
                    best_eval_trial_avg_metric = eval_epoch_results[f'eval_trial_avg_{self.metric}']
                    print(f"epoch: {epoch} best eval loss: {best_eval_loss}")
                    print(f"epoch: {epoch} best eval trial avg {self.metric}: {best_eval_trial_avg_metric}")
                    # save model
                    self.save_model(name="best", epoch=epoch)
                    if self.config.method.model_kwargs.method_name == 'ssl':
                        gt_pred_fig = self.plot_epoch(
                            gt=eval_epoch_results['eval_gt'][0], 
                            preds=eval_epoch_results['eval_preds'][0], epoch=epoch,
                            active_neurons=self.session_active_neurons[0][:5]
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

                print(f"epoch: {epoch} eval loss: {eval_epoch_results['eval_loss']} {self.metric}: {eval_epoch_results[f'eval_trial_avg_{self.metric}']}")

            # save model by epoch
            if epoch % self.config.training.save_every == 0:
                self.save_model(name="epoch", epoch=epoch)

            # plot epoch
            if epoch % self.config.training.save_plot_every_n_epochs == 0:
                if self.config.method.model_kwargs.method_name == 'ssl':

                    gt_pred_fig = self.plot_epoch(
                        gt=eval_epoch_results['eval_gt'][0], 
                        preds=eval_epoch_results['eval_preds'][0], 
                        epoch=epoch,
                        active_neurons=self.session_active_neurons[0][:5]
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
                    self.model.encoder.masker.ratio = 0.6
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
            spike_augmentation=self.config.data.spike_augmentation,
            num_neuron=batch['spikes_data'].shape[2],
            eid=batch['eid'][0]  # each batch consists of data from the same eid
        ) 
    
    def eval_epoch(self):
        self.model.eval()
        eval_loss = 0.
        eval_examples = 0
        session_results = {}
        for num_neuron in self.num_neurons:
            session_results[num_neuron] = {
                "gt": [],
                "preds": []
            }
        if self.eval_dataloader:
            gt, preds = [], []
            with torch.no_grad():  
                for batch in self.eval_dataloader:
                    if self.masking_mode in ["combined", "all"]:
                        masking_mode = random.sample(self.masking_schemes, 1)[0]
                        if masking_mode == 'temporal':
                            self.model.encoder.masker.ratio = 0.3
                        elif masking_mode == 'causal':
                            self.model.encoder.masker.ratio = 0.6
                        else:
                            self.model.encoder.masker.ratio = self.masking_ratio
                    else:
                        masking_mode = self.masking_mode
                    outputs = self._forward_model_outputs(batch, masking_mode)
                    loss = outputs.loss
                    eval_loss += loss.item()
                    eval_examples += outputs.n_examples
                    if self.model_class in ['NDT1', 'iTransformer']:
                        num_neuron = batch['spikes_data'].shape[2]
                    elif self.model_class in ['NDT2', 'STPatch']:
                        num_neuron = outputs.num_neuron
                    if self.config.method.model_kwargs.method_name == 'ssl':
                        session_results[num_neuron]["gt"].append(outputs.targets.clone()[:,:,:num_neuron])
                        session_results[num_neuron]["preds"].append(outputs.preds.clone()[:,:,:num_neuron])
                    else:
                        session_results[num_neuron]["gt"].append(outputs.targets.clone())
                        session_results[num_neuron]["preds"].append(outputs.preds.clone())
                    
            results_list = []
            for idx, num_neuron in enumerate(self.num_neurons):
                _gt = torch.cat(session_results[num_neuron]["gt"], dim=0)
                _preds = torch.cat(session_results[num_neuron]["preds"], dim=0)

                if self.config.method.model_kwargs.loss == "poisson_nll":
                    _preds = torch.exp(_preds)
                elif self.config.method.model_kwargs.loss == "cross_entropy" :
                    _preds = torch.nn.functional.softmax(_preds, dim=1)
                gt.append(_gt)
                preds.append(_preds)

                if len(self.session_active_neurons) < len(self.num_neurons):
                    active_neurons = np.argsort(gt[idx].cpu().numpy().sum((0,1)))[::-1][:50].tolist()
                    self.session_active_neurons.append(active_neurons)
                if self.config.method.model_kwargs.method_name == 'ssl':
                    results = metrics_list(gt = gt[idx][:,:,self.session_active_neurons[idx]].transpose(-1,0),
                                        pred = preds[idx][:,:,self.session_active_neurons[idx]].transpose(-1,0), 
                                        metrics=["r2"], 
                                        device=self.accelerator.device)
                    
                elif self.config.method.model_kwargs.method_name == 'sl':
                    if self.config.method.model_kwargs.clf:
                        results = metrics_list(gt = gt[idx].argmax(1),
                                            pred = preds[idx].argmax(1), 
                                            metrics=[self.metric], 
                                            device=self.accelerator.device)
                    elif self.config.method.model_kwargs.reg:
                        results = metrics_list(gt = gt[idx],
                                            pred = preds[idx],
                                            metrics=[self.metric],
                                            device=self.accelerator.device)
                results_list.append(results[self.metric])

        return {
            "eval_loss": eval_loss/eval_examples,
            f"eval_trial_avg_{self.metric}": np.mean(results_list),
            "eval_gt": gt,
            "eval_preds": preds,
        }
    
    def plot_epoch(self, gt, preds, epoch, active_neurons):
        gt_pred_fig = plot_gt_pred(gt = gt.mean(0).T.cpu().numpy(),
                    pred = preds.mean(0).T.detach().cpu().numpy(),
                    epoch = epoch)
        
        r2_fig = plot_neurons_r2(gt = gt.mean(0),
                pred = preds.mean(0),
                neuron_idx=active_neurons,
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
        
