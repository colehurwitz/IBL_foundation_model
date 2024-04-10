import torch
import numpy as np
import wandb
import os
from utils.utils import move_batch_to_device, metrics_list, plot_gt_pred, plot_neurons_r2

class Trainer():
    def __init__(
            self,
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            optimizer,
            **kwargs
    ):
        # get all the arguments
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer

        # get arguments from kwargs if they exist
        self.log_dir = kwargs.get("log_dir", None)
        self.accelerator = kwargs.get("accelerator", None)
        self.lr_scheduler = kwargs.get("lr_scheduler", None)
        self.config = kwargs.get("config", None)

        self.active_neurons = None

    def train(self):
        best_test_loss = torch.tensor(float('inf'))
        best_eval_trial_avg_r2 = -torch.tensor(float('inf'))
        best_test_trial_avg_r2 = -torch.tensor(float('inf'))
        # train loop
        for epoch in range(self.config.training.num_epochs):
            train_epoch_results = self.train_epoch(epoch)
            eval_epoch_results = self.eval_epoch()
            test_epoch_results = self.test_epoch()
            print(f"epoch: {epoch} train loss: {train_epoch_results['train_loss'] }")
            # if eval_epoch_results dict is not empty
            if eval_epoch_results:
                print(f"epoch: {epoch} eval loss: {eval_epoch_results['eval_loss']}")
                if eval_epoch_results['eval_trial_avg_r2'] > best_eval_trial_avg_r2:
                    best_eval_trial_avg_r2 = eval_epoch_results['eval_trial_avg_r2']
                    print(f"epoch: {epoch} best eval trial avg r2: {best_eval_trial_avg_r2}")
            # if test_epoch_results dict is not empty
            if test_epoch_results:
                if test_epoch_results['test_trial_avg_r2'] > best_test_trial_avg_r2:
                    best_test_trial_avg_r2 = test_epoch_results['test_trial_avg_r2']
                    print(f"epoch: {epoch} best test trial avg r2: {best_test_trial_avg_r2}")
                    # save model
                    self.save_model(name="best", epoch=epoch)
                if test_epoch_results['test_loss'] < best_test_loss:
                    best_test_loss = test_epoch_results['test_loss']
                    print(f"epoch: {epoch} best test loss: {best_test_loss}")
                print(f"epoch: {epoch} test loss: {test_epoch_results['test_loss']} r2: {test_epoch_results['test_trial_avg_r2']}")

            # save model by epoch
            if epoch % self.config.training.save_every == 0:
                self.save_model(name="epoch", epoch=epoch)

            # plot epoch
            if epoch % self.config.training.save_plot_every_n_epochs == 0:
                gt_pred_fig = self.plot_epoch(
                    gt=test_epoch_results['test_gt'], 
                    preds=test_epoch_results['test_preds'], 
                    epoch=epoch, 
                    n_sample_neurons=self.config.training.n_sample_neurons
                )
                if self.config.wandb.use:
                    wandb.log({"gt_pred_fig": wandb.Image(gt_pred_fig['plot_gt_pred']),
                               "r2_fig": wandb.Image(gt_pred_fig['plot_r2'])})
                else:
                    gt_pred_fig['plot_gt_pred'].savefig(os.path.join(self.log_dir, f"gt_pred_fig_{epoch}.png"))
                    gt_pred_fig['plot_r2'].savefig(os.path.join(self.log_dir, f"r2_fig_{epoch}.png"))

            # wandb log
            if self.config.wandb.use:
                wandb.log({"train_loss": train_epoch_results['train_loss'],
                           "test_loss": test_epoch_results['test_loss'],
                           "test_trial_avg_r2": test_epoch_results['test_trial_avg_r2']})
                
        # save last model
        self.save_model(name="last", epoch=epoch)
        
        if self.config.wandb.use:
            wandb.log({"best_test_loss": best_test_loss,
                       "best_test_trial_avg_r2": best_test_trial_avg_r2})
            
    def train_epoch(self, epoch):
        train_loss = 0.
        train_examples = 0
        self.model.train()
        for batch in self.train_dataloader:
            outputs = self._forward_model_outputs(batch)
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
    
    def _forward_model_outputs(self, batch):
        batch = move_batch_to_device(batch, self.accelerator.device)
        return self.model(
            batch['spikes_data'], 
            batch['attention_mask'], 
            batch['spikes_timestamps']
        ) 
        
    def eval_epoch(self):
        # TODO: implement this for decoding
        self.model.eval()
        if self.eval_dataloader:
            pass
        else:
            return None
    
    def test_epoch(self):
        self.model.eval()
        test_loss = 0.
        test_examples = 0
        if self.test_dataloader:
            gt = []
            preds = []
            with torch.no_grad():  
                for batch in self.test_dataloader:
                    outputs = self._forward_model_outputs(batch)
                    loss = outputs.loss
                    test_loss += loss.item()
                    test_examples += outputs.n_examples
                    if self.config.data.patching:
                        gt.append(
                            outputs.targets.clone()
                        )
                        preds.append(
                            outputs.preds.clone()
                        )
                    else:
                        preds.append(outputs.preds.clone())
            
            gt = torch.cat(gt, dim=0)
            preds = torch.cat(preds, dim=0)
        if self.config.method.model_kwargs.loss == "poisson_nll":
            preds = torch.exp(preds)
        if self.active_neurons is None:
            self.active_neurons = np.argsort(gt.cpu().numpy().sum((0,1)))[::-1][:5].tolist()

        if self.config.training.n_sample_neurons is None:
            n_sample_neurons = gt.shape[-1]
        else:
            n_sample_neurons = self.config.training.n_sample_neurons
            
        results = metrics_list(gt = gt.mean(0)[:,:n_sample_neurons][..., self.active_neurons].T,
                               pred = preds.mean(0)[:,:n_sample_neurons][..., self.active_neurons].T, 
                               metrics=["r2"], 
                               device=self.accelerator.device)

        return {
            "test_loss": test_loss/test_examples,
            "test_trial_avg_r2": results['r2'],
            "test_gt": gt,
            "test_preds": preds,
        }
    
    def plot_epoch(self, gt, preds, epoch, n_sample_neurons=None):

        if n_sample_neurons is None:
            n_sample_neurons = gt.shape[-1]
        
        gt_pred_fig = plot_gt_pred(gt = gt.mean(0)[:,:n_sample_neurons].T.cpu().numpy(),
                     pred = preds.mean(0)[:,:n_sample_neurons].T.detach().cpu().numpy(),
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
        print(f"saving model: {name}")
        dict_config = {
            "model": self.model,
            "epoch": epoch,
        }
        torch.save(dict_config, os.path.join(self.log_dir, f"model_{name}.pt"))


