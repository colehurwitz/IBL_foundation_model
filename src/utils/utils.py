import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.metric_utils import r2_score

def set_seed(seed):
    # set seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print('seed set to {}'.format(seed))

def move_batch_to_device(batch, device):
    # if batch values are tensors, move them to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch

def plot_gt_pred(gt, pred, epoch=0):
    # plot Ground Truth and Prediction in the same figur
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("Ground Truth")
    im1 = ax1.imshow(gt, aspect='auto', cmap='binary')
    
    ax2.set_title("Prediction")
    im2 = ax2.imshow(pred, aspect='auto', cmap='binary')
    
    # add colorbar
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)

    fig.suptitle("Epoch: {}".format(epoch))
    return fig

def plot_r2(gt, pred, r2, epoch=0, neuron_idx=0):
    # plot line of gt and pred in different colors
    fig, ax = plt.subplots()
    ax.plot(gt, label="Ground Truth", color="blue")
    ax.plot(pred, label="Prediction", color="red")
    ax.set_title("R2: {:.4f}".format(r2))
    ax.legend()
    # x label
    ax.set_xlabel("Time")
    # y label
    ax.set_ylabel("Rate")
    fig.suptitle("Epoch: {}, Neuron: {}".format(epoch, neuron_idx))
    return fig


# metrics list, return different metrics results
def metrics_list(gt, pred, metrics=["r2", "mse", "mae"], device="cpu"):
    results = {}
    if "r2" in metrics:
        r2 = r2_score(gt, pred)
        results["r2"] = r2
    if "mse" in metrics:
        mse = torch.mean((gt - pred) ** 2)
        results["mse"] = mse
    if "mae" in metrics:
        mae = torch.mean(torch.abs(gt - pred))
        results["mae"] = mae
    return results


# plot the GT rate and the Prediction rate (after exp). TODO: add the spikes. unavailable now.
def plot_rate_and_spike(gt, pred, epoch=0, eps=1e-7):
    # find the most active neurons
    row_sum = np.sum(gt, axis=1)
    idxs_top6 = np.argsort(row_sum)[-6:]
    pred_top6 = pred[idxs_top6, :]
    gt_top6 = gt[idxs_top6, :]

    fig, axes = plt.subplots(2, 3, figsize=(24, 6))

    for i, ax in enumerate(axes.flat):
        ax.set_ylim(-2, 3)
        # normalized TODO: change the var to be more serious
        y1 = (gt_top6[i, :] - np.mean(gt_top6[i, :])) / (np.sqrt(np.var(gt_top6[i, :])) + eps)
        y2 = (pred_top6[i, :] - np.mean(pred_top6[i, :])) / (np.sqrt(np.var(pred_top6[i, :])) + eps)
        ax.plot(y1, 'grey')
        ax.plot(y2, 'blue')
        ax.set_title(f"Neuron #{idxs_top6[i]}")

    fig.suptitle(f"Epoch: {epoch}")
    return fig


# plot the trail average rate and spike counts
def plot_avg_rate_and_spike(output, epoch=0):  # output is the list of batches (n_batch, 2, bs, seq_len, n_neurons)

    pred_avg = np.zeros(output[0][0][0].detach().cpu().numpy().shape)
    gt_avg = np.zeros(output[0][0][0].detach().cpu().numpy().shape)
    n_trials = 0

    for i, batch in enumerate(output):
        pred = batch[0].detach().cpu().numpy()
        gt = batch[1].detach().cpu().numpy()
        pred_avg = pred_avg + np.sum(pred, axis=0)
        gt_avg = gt_avg + np.sum(gt, axis=0)
        n_trials = n_trials + pred.shape[0]
    pred_avg = pred_avg.T / n_trials
    gt_avg = gt_avg.T / n_trials

    # find the most active neurons
    row_sum = np.sum(gt_avg, axis=1)
    idxs_top6 = np.argsort(row_sum)[-6:]
    pred_avg_top6 = pred_avg[idxs_top6, :]
    gt_avg_top6 = gt_avg[idxs_top6, :]

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))

    for i, ax in enumerate(axes.flat):
        ax.set_ylim(0, 2)
        y1 = pred_avg_top6[i, :]
        y2 = gt_avg_top6[i, :]
        ax.plot(y1, 'blue')
        ax.plot(y2, 'grey')
        ax.set_title(f'Neuron: #{idxs_top6[i]}')

    fig.suptitle(f"Epoch: {epoch},Avg from {n_trials} trials")
    return fig
