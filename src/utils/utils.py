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
    