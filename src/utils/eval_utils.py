from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
from accelerate import Accelerator
from src.loader.make_loader import make_loader
from src.utils.dataset_utils import split_both_dataset
from src.utils.utils import set_seed, move_batch_to_device, plot_gt_pred, metrics_list, plot_avg_rate_and_spike, \
    plot_rate_and_spike
from src.utils.config_utils import config_from_kwargs, update_config
from src.models.ndt1 import NDT1
from src.models.stpatch import STPatch
from models.itransformer import iTransformer
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import r2_score
from scipy.special import gammaln
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from sklearn.cluster import SpectralClustering
import os

NAME2MODEL = {"NDT1": NDT1, "STPatch": STPatch, "iTransformer": iTransformer}

import logging

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------------
# Model/Dataset Loading and Configuration
# --------------------------------------------------------------------------------------------------

def load_model_data_local(**kwargs):
    model_config = kwargs['model_config']
    trainer_config = kwargs['trainer_config']
    model_path = kwargs['model_path']
    dataset_path = kwargs['dataset_path']
    test_size = kwargs['test_size']
    seed = kwargs['seed']

    # set seed
    set_seed(seed)

    # load the model
    config = config_from_kwargs({"model": f"include:{model_config}"})
    config = update_config(model_config, config)
    config = update_config(trainer_config, config)

    accelerator = Accelerator()

    model_class = NAME2MODEL[config.model.model_class]
    model = model_class(config.model, **config.method.model_kwargs)
    model = accelerator.prepare(model)

    model.load_state_dict(torch.load(model_path)['model'].state_dict())

    # load the dataset
    r_dataset = load_from_disk(dataset_path)
    dataset = r_dataset.train_test_split(test_size=test_size, seed=seed)['test']
    try:
        bin_size = dataset['binsize'][0]
    except:
        bin_size = dataset['bin_size'][0]
    print(f'bin size: {bin_size}')

    # TODO: update the loader to adapt other models (e.g., patching for NDT2)
    dataloader = make_loader(
        dataset,
        batch_size=10000,
        pad_to_right=True,
        pad_value=-1.,
        bin_size=bin_size,
        max_time_length=config.data.max_time_length,
        max_space_length=config.data.max_space_length,
        dataset_name=config.data.dataset_name,
        shuffle=False
    )

    # check the shape of the dataset
    for batch in dataloader:
        print('spike data shape: {}'.format(batch['spikes_data'].shape))
        break

    return model, accelerator, dataset, dataloader


# --------------------------------------------------------------------------------------------------
# Evaluation
# 1. Co-smoothing_r2 (R2 and shuqi's plot) TODO: add more behaviors, add n!=1
# 2. Co-smoothing_bps (co-bps, adapted from NLB repo) TODO: how to choose the held-out neurons?
# 3. R2 scatter plot
# 4. Forward-prediction_r2 (R2 and shuqi's plot w/o PSTH)
# 5. Forward-prediction_bps (co-bps, adapted from NLB repo)
# --------------------------------------------------------------------------------------------------

def co_smoothing_r2(
        model,
        accelerator,
        test_dataloader,
        test_dataset,
        n=1,
        **kwargs
):
    assert n == 1, 'only support n=1 now'

    uuids_list = np.array(test_dataset['cluster_uuids'][0])
    region_list = np.array(test_dataset['cluster_regions'])[0, :]

    for batch in test_dataloader:
        break

    r2_result_list = []

    # loop through all the neurons
    for n_i in range(batch['spikes_data'].shape[-1]):

        # validate the model on test set
        model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                batch = move_batch_to_device(batch, accelerator.device)
                mask_result = heldout_mask(
                    batch['spikes_data'],
                    mode='manual',
                    heldout_idxs=np.array([n_i])
                )

                outputs = model(
                    mask_result['spikes'],
                    batch['time_attn_mask'],
                    batch['space_attn_mask'],
                    batch['spikes_timestamps'],
                    batch['spikes_spacestamps']
                )

        # exponential the poisson rates
        outputs.preds = torch.exp(outputs.preds)

        # got the numpy array for gt and pred
        gt_spikes = batch['spikes_data'].detach().cpu().numpy()
        pred_spikes = outputs.preds.detach().cpu().numpy()

        # prepare the condition matrix
        b_list = []

        # choice
        choice = np.array(test_dataset['choice'])
        choice = np.tile(np.reshape(choice, (choice.shape[0], 1)), (1, 100))
        b_list.append(choice)

        # reward
        reward = np.array(test_dataset['reward'])
        reward = np.tile(np.reshape(reward, (reward.shape[0], 1)), (1, 100))
        b_list.append(reward)

        # block
        block = np.array(test_dataset['block'])
        block = np.tile(np.reshape(block, (block.shape[0], 1)), (1, 100))
        b_list.append(block)

        behavior_set = np.stack(b_list, axis=-1)

        # Settings for validation
        X = behavior_set  # [#trials, #timesteps, #variables]
        ys = gt_spikes  # [#trials, #timesteps, #neurons]
        y_preds = pred_spikes  # [#trials, #timesteps, #neurons]

        var_name2idx = {'block': [2],
                        'choice': [0],
                        'reward': [1],
                        'wheel': [3],
                        }

        var_value2label = {'block': {(0.2,): "p(left)=0.2",
                                     (0.5,): "p(left)=0.5",
                                     (0.8,): "p(left)=0.8", },
                           'choice': {(-1.0,): "right",
                                      (1.0,): "left"},
                           'reward': {(0.,): "no reward",
                                      (1.,): "reward", }}

        var_tasklist = ['block', 'choice', 'reward']
        var_behlist = []

        # choose the neuron to plot (idx_top / held_out_idx / ...)
        idxs = mask_result['heldout_idxs']

        method_name = kwargs['method_name']
        for i in range(idxs.shape[0]):
            _r2_psth, _r2_trial = viz_single_cell(X, ys[:, :, idxs[i]], y_preds[:, :, idxs[i]],
                                                  var_name2idx, var_tasklist, var_value2label, var_behlist,
                                                  subtract_psth=kwargs['subtract'],
                                                  aligned_tbins=kwargs['onset_alignment'],
                                                  neuron_idx=uuids_list[idxs[i]][:4],
                                                  neuron_region=region_list[idxs[i]],
                                                  method=method_name, save_path=kwargs['save_path'])
            r2_result_list.append(np.array([_r2_psth, _r2_trial]))
            plt.show()

    r2_all = np.array(r2_result_list)
    np.save(os.path.join(kwargs['save_path'], f'r2.npy'), r2_all)


def co_smoothing_bps(
        model,
        accelerator,
        test_dataloader,
        mode='per_neuron',              # manual / active / region / per_neuron / etc (TODO)
        held_out_list=None,             # list for manual mode
):
    for batch in test_dataloader:
        break

    if mode == 'per_neuron':
        bps_result_list = []

        # loop through all the neurons
        for n_i in tqdm(range(batch['spikes_data'].shape[-1]), desc='neuron'):
        # for n_i in tqdm(range(200), desc='neuron'):

            # validate the model on test set
            model.eval()
            with torch.no_grad():
                for batch in test_dataloader:
                    batch = move_batch_to_device(batch, accelerator.device)
                    mask_result = heldout_mask(
                        batch['spikes_data'],
                        mode='manual',
                        heldout_idxs=np.array([n_i])
                    )

                    outputs = model(
                        mask_result['spikes'],
                        batch['time_attn_mask'],
                        batch['space_attn_mask'],
                        batch['spikes_timestamps'],
                        batch['spikes_spacestamps']
                    )

            # exponential the poisson rates
            outputs.preds = torch.exp(outputs.preds)

            # got the numpy array for gt and pred
            gt_spikes = batch['spikes_data'].detach().cpu().numpy()
            pred_spikes = outputs.preds.detach().cpu().numpy()

            gt_held_out = gt_spikes[:, :, [n_i]]
            pred_held_out = pred_spikes[:, :, [n_i]]

            bps = bits_per_spike(pred_held_out, gt_held_out)
            # print(bps)
            if np.isinf(bps):
                continue
            bps_result_list.append(bps)

        bps_all = np.array(bps_result_list)
        bps_mean = np.mean(bps_all)
        bps_std = np.std(bps_all)
        plt.hist(bps_all, bins=30, alpha=0.75, color='red', edgecolor='black')
        plt.xlabel('bits per spike')
        plt.ylabel('count')
        plt.title('Co-bps distribution\n mean: {:.2f}, std: {:.2f}\n # non-zero neuron: {}'.format(bps_mean, bps_std, len(bps_all)))
        plt.show()        

    else:
        raise NotImplementedError('mode not implemented')


def compare_R2_scatter(**kwargs):
    A_path = kwargs['A_path'],
    B_path = kwargs['B_path'],
    A_name = kwargs['A_name'],
    B_name = kwargs['B_name'],

    A_r2 = np.load(os.path.join(A_path, 'r2.npy'))
    B_r2 = np.load(os.path.join(B_path, 'r2.npy'))

    A_psth = A_r2[:, 0]
    B_psth = B_r2[:, 0]
    A_psth[A_psth < 0] = 0
    B_psth[B_psth < 0] = 0

    A_r2 = A_r2[:, 1]
    B_r2 = B_r2[:, 1]
    A_r2[A_r2 < 0] = 0
    B_r2[B_r2 < 0] = 0

    line_x = np.linspace(0, 1, 100)
    line_y = line_x

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].scatter(A_psth, B_psth, alpha=0.9, s=1)
    axes[0].plot(line_x, line_y, color='black', lw=1)
    axes[0].set_xlabel(A_name)
    axes[0].set_ylabel(B_name)
    axes[0].set_title('R2_PSTH')

    axes[1].scatter(A_r2, B_r2, alpha=0.9, s=1)
    axes[1].plot(line_x, line_y, color='black', lw=1)
    axes[1].set_xlabel(A_name)
    axes[1].set_ylabel(B_name)
    axes[1].set_title('R2')


def forward_prediction_r2(
    model,
    accelerator,
    test_dataloader,
    test_dataset,
    held_out_list=None,
    **kwargs
):
    uuids_list = np.array(test_dataset['cluster_uuids'][0])
    region_list = np.array(test_dataset['cluster_regions'])[0, :]
    
    for batch in test_dataloader:
        break

    # validate the model on test set
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            batch = move_batch_to_device(batch, accelerator.device)
            mask_result = heldout_mask(
                batch['spikes_data'],
                mode='forward-pred',
                heldout_idxs=np.array(held_out_list)
            )
            outputs = model(
                mask_result['spikes'],
                batch['time_attn_mask'],
                batch['space_attn_mask'],
                batch['spikes_timestamps'],
                batch['spikes_spacestamps']
            )

    # exponential the poisson rates
    outputs.preds = torch.exp(outputs.preds)

    # got the numpy array for gt and pred
    gt_spikes = batch['spikes_data'].detach().cpu().numpy()
    pred_spikes = outputs.preds.detach().cpu().numpy()

    r2_result_list = []
    # loop through all the neurons
    for n_i in tqdm(range(batch['spikes_data'].shape[-1]), desc='neuron'):
        gt, pred = gt_spikes[:,:,n_i], pred_spikes[:,:,n_i]   
        r2 = viz_single_cell_no_psth(
            gt, pred, 
            neuron_idx=uuids_list[idxs[i]][:4],
            neuron_region=region_list[idxs[i]],
            method=method_name, save_path=kwargs['save_path']
        )
        r2_result_list.append(r2)
        plt.show()

    r2_all = np.array(r2_result_list)
    np.save(os.path.join(kwargs['save_path'], f'r2.npy'), r2_all)
    

def forward_prediction_bps(
    model,
    accelerator,
    test_dataloader,
    held_out_list=None,  
    **kwargs
)
    for batch in test_dataloader:
        break

    # validate the model on test set
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            batch = move_batch_to_device(batch, accelerator.device)
            mask_result = heldout_mask(
                batch['spikes_data'],
                mode='forward-pred',
                heldout_idxs=np.array(held_out_list)
            )

            outputs = model(
                mask_result['spikes'],
                batch['time_attn_mask'],
                batch['space_attn_mask'],
                batch['spikes_timestamps'],
                batch['spikes_spacestamps']
            )

    # exponential the poisson rates
    outputs.preds = torch.exp(outputs.preds)

    # got the numpy array for gt and pred
    gt_spikes = batch['spikes_data'].detach().cpu().numpy()
    pred_spikes = outputs.preds.detach().cpu().numpy()

    gt_held_out = gt_spikes[:, held_out_list, :]
    pred_held_out = pred_spikes[:, held_out_list, :]

    bps_result_list = []
    # loop through all the neurons
    for n_i in tqdm(range(batch['spikes_data'].shape[-1]), desc='neuron'): 
        bps = bits_per_spike(pred_held_out[:,:,[n_i]], gt_held_out[:,:,[n_i]])
        if np.isinf(bps):
            continue
        bps_result_list.append(bps)

    bps_all = np.array(bps_result_list)
    bps_mean = np.mean(bps_all)
    bps_std = np.std(bps_all)
    plt.hist(bps_all, bins=30, alpha=0.75, color='red', edgecolor='black')
    plt.xlabel('bits per spike')
    plt.ylabel('count')
    plt.title('Co-bps distribution\n mean: {:.2f}, std: {:.2f}\n # non-zero neuron: {}'.format(bps_mean, bps_std, len(bps_all)))
    plt.show()
    
    np.save(os.path.join(kwargs['save_path'], f'co-bps.npy'), bps_all)


# --------------------------------------------------------------------------------------------------
# helper functions
# --------------------------------------------------------------------------------------------------

def heldout_mask(
        spike_data,                     # (K, T, N)
        mode='manual',                  # manual / active / region / per_neuron / etc (TODO)
        heldout_idxs=np.array([]),      # list for manual mode
        n_active=1,                     # n_neurons for most-active mode
        region=None                     # list for region mode
):
    mask = torch.ones(spike_data.shape).to(spike_data.device)

    if mode == 'manual':
        hd = heldout_idxs
        mask[:, :, hd] = 0

    elif mode == 'most':
        _act = spike_data.detach().cpu().numpy()
        _act = np.mean(_act, axis=(0, 1))
        act_idx = np.argsort(_act)
        hd = np.array(act_idx[-n_active:])
        mask[:, :, hd] = 0

    elif mode == 'region':
        raise NotImplementedError('region mask not implemented')

    elif mode == 'forward-pred':
        hd = heldout_idxs
        mask[:, hd, :] = 0

    spike_data_masked = spike_data * mask

    return {"spikes": spike_data_masked, "heldout_idxs": hd}


# --------------------------------------------------------------------------------------------------
# copied from NLB repo
# standard evaluation metrics
# --------------------------------------------------------------------------------------------------

def neg_log_likelihood(rates, spikes, zero_warning=True):
    """Calculates Poisson negative log likelihood given rates and spikes.
    formula: -log(e^(-r) / n! * r^n)
           = r - n*log(r) + log(n!)

    Parameters
    ----------
    rates : np.ndarray
        numpy array containing rate predictions
    spikes : np.ndarray
        numpy array containing true spike counts
    zero_warning : bool, optional
        Whether to print out warning about 0 rate
        predictions or not

    Returns
    -------
    float
        Total negative log-likelihood of the data
    """
    assert (
            spikes.shape == rates.shape
    ), f"neg_log_likelihood: Rates and spikes should be of the same shape. spikes: {spikes.shape}, rates: {rates.shape}"

    if np.any(np.isnan(spikes)):
        mask = np.isnan(spikes)
        rates = rates[~mask]
        spikes = spikes[~mask]

    assert not np.any(np.isnan(rates)), "neg_log_likelihood: NaN rate predictions found"

    assert np.all(rates >= 0), "neg_log_likelihood: Negative rate predictions found"
    if np.any(rates == 0):
        if zero_warning:
            logger.warning(
                "neg_log_likelihood: Zero rate predictions found. Replacing zeros with 1e-9"
            )
        rates[rates == 0] = 1e-9

    result = rates - spikes * np.log(rates) + gammaln(spikes + 1.0)
    # print('nll_score', np.sum(result))
    # print('rate', rates.reshape(-1, rates.shape[1]*rates.shape[2]), '\nspikes', spikes.reshape(-1, spikes.shape[1]*spikes.shape[2]), '\nresult', result.reshape(-1, result.shape[1]*result.shape[2]))
    # print(rates.shape, spikes.shape, result.shape)
    return np.sum(result)


def bits_per_spike(rates, spikes):
    """Computes bits per spike of rate predictions given spikes.
    Bits per spike is equal to the difference between the log-likelihoods (in base 2)
    of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    Parameters
    ----------
    rates : np.ndarray
        3d numpy array containing rate predictions
    spikes : np.ndarray
        3d numpy array containing true spike counts

    Returns
    -------
    float
        Bits per spike of rate predictions
    """
    nll_model = neg_log_likelihood(rates, spikes)
    null_rates = np.tile(
        np.nanmean(spikes, axis=tuple(range(spikes.ndim - 1)), keepdims=True),
        spikes.shape[:-1] + (1,),
    )
    nll_null = neg_log_likelihood(null_rates, spikes, zero_warning=False)
    # print(np.nansum(spikes))
    return (nll_null - nll_model) / np.nansum(spikes) / np.log(2)


# --------------------------------------------------------------------------------------------------
# single neuron plot functions
# --------------------------------------------------------------------------------------------------

"""
:X: [n_trials, n_timesteps, n_variables]
:y: [n_trials, n_timesteps] (in Hz)
:y_pred: [n_trials, n_timesteps] (in Hz)
:var_tasklist: for each task variable in var_tasklists, compute PSTH
:var_name2idx: for each task variable in var_tasklists, the corresponding index of X
:var_value2label:
:aligned_tbins: reference time steps to annotate. 
"""


def plot_psth(X, y, y_pred, var_tasklist, var_name2idx, var_value2label,
              aligned_tbins=[],
              axes=None, legend=False, neuron_idx='', neuron_region=''):
    if axes is None:
        nrows = 1;
        ncols = len(var_tasklist)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2 * nrows))

    for ci, var in enumerate(var_tasklist):
        ax = axes[ci]
        psth_xy = compute_all_psth(X, y, var_name2idx[var])
        psth_pred_xy = compute_all_psth(X, y_pred, var_name2idx[var])
        for _i, _x in enumerate(psth_xy.keys()):
            psth = psth_xy[_x]
            psth_pred = psth_pred_xy[_x]
            ax.plot(psth,
                    color=plt.get_cmap('tab10')(_i),
                    linewidth=3, alpha=0.3, label=f"{var_value2label[var][tuple(_x)]}")
            ax.plot(psth_pred,
                    color=plt.get_cmap('tab10')(_i),
                    linestyle='--')
            ax.set_xlabel("Time bin")
            if ci == 0:
                ax.set_ylabel("Neural activity")
            else:
                ax.sharey(axes[0])
        _add_baseline(ax, aligned_tbins=aligned_tbins)
        if legend:
            ax.legend()
            ax.set_title(f"{var}")

    # compute PSTH for task_contingency
    idxs_psth = np.concatenate([var_name2idx[var] for var in var_tasklist])
    psth_xy = compute_all_psth(X, y, idxs_psth)
    psth_pred_xy = compute_all_psth(X, y_pred, idxs_psth)
    r2_psth = compute_R2_psth(psth_xy, psth_pred_xy, clip=False)
    r2_single_trial = compute_R2_main(y.reshape(-1, 1), y_pred.reshape(-1, 1), clip=False)[0]
    '''
    axes[-1].annotate(f'PSTH R2: {r2_psth:.2f}'+"\n"+f"#conds: {len(psth_xy.keys())}", 
                        xy=(y.shape[1], 0), 
                        xytext=(y.shape[1]+20, 0), 
                        ha='left', 
                        rotation=90)
    '''
    axes[0].set_ylabel(
        f'Neuron: #{neuron_idx[:4]} \n PSTH R2: {r2_psth:.2f} \n Avg_SingleTrial R2: {r2_single_trial:.2f}')

    for ax in axes:
        # ax.axis('off')
        ax.spines[['right', 'top']].set_visible(False)
        # ax.set_frame_on(False)
        # ax.tick_params(bottom=False, left=False)
    plt.tight_layout()

    return r2_psth, r2_single_trial


"""
:X: [n_trials, n_timesteps, n_variables]
:y: [n_trials, n_timesteps] (in Hz)
:y_pred: [n_trials, n_timesteps] (in Hz)
:var_tasklist: variables used for computing the task-condition-averaged psth if subtract_psth=='task'
:var_name2idx:
:var_tasklist: variables to be plotted in the single-trial behavior
:subtract_psth: 
    - None: no subtraction
    - "task": subtract task-condition-averaged psth
    - "global": subtract global-averaged psth
:aligned_tbins: reference time steps to annotate. 
:nclus, n_neighbors: hyperparameters for spectral_clustering
:cmap, vmax_perc, vmin_perc: parameters used when plotting the activity and behavior
"""


def plot_single_trial_activity(X, y, y_pred,
                               var_name2idx,
                               var_behlist,
                               var_tasklist, subtract_psth="task",
                               aligned_tbins=[],
                               n_clus=8, n_neighbors=5, n_pc=32, clusby='y_pred',
                               cmap='bwr', vmax_perc=90, vmin_perc=10,
                               axes=None):
    if axes is None:
        ncols = 1;
        nrows = 2 + len(var_behlist) + 1 + 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 3 * nrows))

    ### get the psth-subtracted y
    if subtract_psth is None:
        pass
    elif subtract_psth == "task":
        idxs_psth = np.concatenate([var_name2idx[var] for var in var_tasklist])
        psth_xy = compute_all_psth(X, y, idxs_psth)
        psth_pred_xy = compute_all_psth(X, y_pred, idxs_psth)
        y_psth = np.asarray(
            [psth_xy[tuple(x)] for x in X[:, 0, idxs_psth]])  # (K, T) predict the neural activity with psth
        y_predpsth = np.asarray(
            [psth_pred_xy[tuple(x)] for x in X[:, 0, idxs_psth]])  # (K, T) predict the neural activity with psth
        y = y - y_psth  # (K, T)
        y_pred = y_pred - y_predpsth  # (K, T)
    elif subtract_psth == "global":
        y_psth = np.mean(y, 0)
        y_predpsth = np.mean(y_pred, 0)
        y = y - y_psth  # (K, T)
        y_pred = y_pred - y_predpsth  # (K, T)
    else:
        assert False, "Unknown subtract_psth, has to be one of: task, global. \'\'"
    y_residual = (y_pred - y)  # (K, T), residuals of prediction
    idxs_behavior = np.concatenate(([var_name2idx[var] for var in var_behlist])) if len(var_behlist) > 0 else []
    X_behs = X[:, :, idxs_behavior]

    ### plot single-trial activity
    # arange the trials by unsupervised clustering labels
    # model = Rastermap(n_clusters=n_clus, # None turns off clustering and sorts single neurons
    #               n_PCs=n_pc, # use fewer PCs than neurons
    #               locality=0.15, # some locality in sorting (this is a value from 0-1)
    #               time_lag_window=15, # use future timepoints to compute correlation
    #               grid_upsample=0, # 0 turns off upsampling since we're using single neurons
    #             )
    # if clusby == 'y_pred':
    #     clustering = model.fit(y_pred)
    # elif clusby == 'y':
    #     clustering = model.fit(y)
    # else:
    #     assert False, "invalid clusby"
    # t_sort = model.isort

    clustering = SpectralClustering(n_clusters=n_clus, n_neighbors=n_neighbors,
                                    affinity='nearest_neighbors',
                                    assign_labels='discretize',
                                    random_state=0)
    if clusby == 'y_pred':
        clustering = clustering.fit(y_pred)
    elif clusby == 'y':
        clustering = clustering.fit(y)
    else:
        assert False, "invalid clusby"
    t_sort = np.argsort(clustering.labels_)

    for ri, (toshow, label, ax) in enumerate(zip([y, y_pred, X_behs, y_residual],
                                                 [f"obs. act. \n (subtract_psth={subtract_psth})",
                                                  f"pred. act. \n (subtract_psth={subtract_psth})",
                                                  var_behlist,
                                                  "residual act."],
                                                 [axes[0], axes[1], axes[2:-2], axes[-2]])):
        if ri <= 1:
            # plot obs./ predicted activity
            vmax = np.percentile(y_pred, vmax_perc)
            vmin = np.percentile(y_pred, vmin_perc)
            raster_plot(toshow[t_sort], vmax, vmin, True, label, ax,
                        cmap=cmap,
                        aligned_tbins=aligned_tbins)
        elif ri == 2:
            # plot behavior
            for bi in range(len(var_behlist)):
                ts_ = toshow[:, :, bi][t_sort]
                vmax = np.percentile(ts_, vmax_perc)
                vmin = np.percentile(ts_, vmin_perc)
                raster_plot(ts_, vmax, vmin, True, label[bi], ax[bi],
                            cmap=cmap,
                            aligned_tbins=aligned_tbins)
        elif ri == 3:
            # plot residual activity
            vmax = np.percentile(toshow, vmax_perc)
            vmin = np.percentile(toshow, vmin_perc)
            raster_plot(toshow[t_sort], vmax, vmin, True, label, ax,
                        cmap=cmap,
                        aligned_tbins=aligned_tbins)

    ### plot single-trial activity
    # re-arrange the trials
    clustering = SpectralClustering(n_clusters=n_clus, n_neighbors=n_neighbors,
                                    affinity='nearest_neighbors',
                                    assign_labels='discretize',
                                    random_state=0).fit(y_residual)
    t_sort_rd = np.argsort(clustering.labels_)
    # model = Rastermap(n_clusters=n_clus, n_PCs=n_pc, locality=0.15, time_lag_window=15, grid_upsample=0,).fit(y_residual)
    # t_sort_rd = model.isort
    raster_plot(y_residual[t_sort_rd], np.percentile(y_residual, vmax_perc), np.percentile(y_residual, vmin_perc), True,
                "residual act. (re-clustered)", axes[-1])

    plt.tight_layout()


"""
This script generates a plot to examine the (single-trial) fitting of a single neuron.
:X: behavior matrix of the shape [n_trials, n_timesteps, n_variables]. 
:y: true neural activity matrix of the shape [n_trials, n_timesteps] 
:ypred: predicted activity matrix of the shape [n_trials, n_timesteps] 
:var_name2idx: dictionary mapping feature names to their corresponding index of the 3-rd axis of the behavior matrix X. e.g.: {"choice": [0], "wheel": [1]}
:var_tasklist: *static* task variables used to form the task condition and compute the psth. e.g.: ["choice"]
:var_value2label: dictionary mapping values in X to their corresponding readable labels (only required for static task variables). e.g.: {"choice": {1.: "left", -1.: "right"}}
:var_behlist: *dynamic* behavior variables. e.g., ["wheel"]
:subtract_psth: 
    - None: no subtraction
    - "task": subtract task-condition-averaged psth
    - "global": subtract global-averaged psth
:algined_tbins: reference time steps to annotate in the plot. 
"""


def viz_single_cell(X, y, y_pred, var_name2idx, var_tasklist, var_value2label, var_behlist,
                    subtract_psth="task", aligned_tbins=[], clusby='y_pred', neuron_idx='', neuron_region='', method='',
                    save_path='figs'):
    nrows = 8
    plt.figure(figsize=(8, 2 * nrows))

    ### plot psth
    axes_psth = [plt.subplot(nrows, len(var_tasklist), k + 1) for k in range(len(var_tasklist))]
    r2_psth, r2_trial = plot_psth(X, y, y_pred,
                                  var_tasklist=var_tasklist,
                                  var_name2idx=var_name2idx,
                                  var_value2label=var_value2label,
                                  aligned_tbins=aligned_tbins,
                                  axes=axes_psth, legend=True, neuron_idx=neuron_idx, neuron_region=neuron_region)

    ### plot the psth-subtracted activity
    axes_single = [plt.subplot(nrows, 1, k) for k in range(2, 2 + 2 + len(var_behlist) + 2)]
    plot_single_trial_activity(X, y, y_pred,
                               var_name2idx,
                               var_behlist,
                               var_tasklist, subtract_psth=subtract_psth,
                               aligned_tbins=aligned_tbins,
                               clusby=clusby,
                               axes=axes_single)

    # print((neuron_region, neuron_idx, r2, method))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f'{neuron_region}_{neuron_idx}_{r2_trial:.2f}_{method}.png'))
    plt.tight_layout();
    # plt.show()

    return r2_psth, r2_trial
    

def viz_single_cell_no_psth(
    gt, pred, neuron_idx, neuron_region, method, save_path
):
    r2 = 0
    for _ in range(len(gt)):
        r2 += r2_score(gt, pred)
    r2 /= len(gt)
    
    y = gt - gt.mean(0)
    y_pred = pred - pred.mean(0)
    y_resid = y - y_pred
    
    vmin_perc, vmax_perc = 10, 90 
    vmax = np.percentile(y_pred, vmax_perc)
    vmin = np.percentile(y_pred, vmin_perc)
    
    toshow = [y, y_pred, y_resid]
    resid_vmax = np.percentile(toshow, vmax_perc)
    resid_vmin = np.percentile(toshow, vmin_perc)
    
    N = len(y)
    y_labels = ['obs.', 'pred.', 'resid.']

    fig, axes = plt.subplots(3, 1, figsize=(8, 7))
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im1 = axes[0].imshow(y, aspect='auto', cmap='bwr', norm=norm)
    cbar = plt.colorbar(im1, pad=0.02, shrink=.6)
    cbar.ax.tick_params(rotation=90)
    axes[0].set_title(f' R2: {r2:.3f}')
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im2 = axes[1].imshow(y_pred, aspect='auto', cmap='bwr', norm=norm)
    cbar = plt.colorbar(im2, pad=0.02, shrink=.6)
    cbar.ax.tick_params(rotation=90)
    norm = colors.TwoSlopeNorm(vmin=resid_vmin, vcenter=0, vmax=resid_vmax)
    im3 = axes[2].imshow(y_resid, aspect='auto', cmap='bwr', norm=norm)
    cbar = plt.colorbar(im3, pad=0.02, shrink=.6)
    cbar.ax.tick_params(rotation=90)
    
    for i, ax in enumerate(axes):
        ax.set_ylabel(f"{y_labels[i]}"+f"\n(#trials={N})")
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.spines[['left','bottom', 'right', 'top']].set_visible(False)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f'{neuron_region}_{neuron_idx}_{r2:.2f}_{method}.png'))
    plt.tight_layout();
    return r2


def _add_baseline(ax, aligned_tbins=[40]):
    for tbin in aligned_tbins:
        ax.axvline(x=tbin - 1, c='k', alpha=0.2)
    # ax.axhline(y=0., c='k', alpha=0.2)


def raster_plot(ts_, vmax, vmin, whether_cbar, ylabel, ax,
                cmap='bwr',
                aligned_tbins=[40]):
    N, T = ts_.shape
    im = ax.imshow(ts_, aspect='auto', cmap=cmap, vmax=vmax, vmin=vmin)
    for tbin in aligned_tbins:
        ax.annotate('',
                    xy=(tbin - 1, N),
                    xytext=(tbin - 1, N + 10),
                    ha='center',
                    va='center',
                    arrowprops={'arrowstyle': '->', 'color': 'r'})
    if whether_cbar:
        cbar = plt.colorbar(im, pad=0.01, shrink=.6)
        cbar.ax.tick_params(rotation=90)
    if not (ylabel is None):
        ax.set_ylabel(f"{ylabel}" + f"\n(#trials={N})")
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.spines[['left', 'bottom', 'right', 'top']].set_visible(False)
        pass
    else:
        ax.axis('off')


"""
- X, y should be nparray with
    - X: [K,T,ncoef]
    - y: [K,T,N] or [K,T]
- axis and value should be list
- return: nparray [T, N] or [T]
"""


def compute_PSTH(X, y, axis, value):
    trials = np.all(X[:, 0, axis] == value, axis=-1)
    return y[trials].mean(0)


def compute_all_psth(X, y, idxs_psth):
    uni_vs = np.unique(X[:, 0, idxs_psth], axis=0)  # get all the unique task-conditions
    psth_vs = {};
    for v in uni_vs:
        # compute separately for true y and predicted y
        _psth = compute_PSTH(X, y,
                             axis=idxs_psth, value=v)  # (T)
        psth_vs[tuple(v)] = _psth
    return psth_vs


"""
psth_xy/ psth_pred_xy: {tuple(x): (T) or (T,N)}
return a float or (N) array
"""


def compute_R2_psth(psth_xy, psth_pred_xy, clip=True):
    psth_xy_array = np.array([psth_xy[x] for x in psth_xy])
    psth_pred_xy_array = np.array([psth_pred_xy[x] for x in psth_xy])
    K, T = psth_xy_array.shape[:2]
    psth_xy_array = psth_xy_array.reshape((K * T, -1))
    psth_pred_xy_array = psth_pred_xy_array.reshape((K * T, -1))
    r2s = [r2_score(psth_xy_array[:, ni], psth_pred_xy_array[:, ni]) for ni in range(psth_xy_array.shape[1])]
    r2s = np.array(r2s)
    # # compute r2 along dim 0
    # r2s = [r2_score(psth_xy[x], psth_pred_xy[x], multioutput='raw_values') for x in psth_xy]
    if clip:
        r2s = np.clip(r2s, 0., 1.)
    # r2s = np.mean(r2s, 0)
    if len(r2s) == 1:
        r2s = r2s[0]
    return r2s


def compute_R2_main(y, y_pred, clip=True):
    """
    :y: (K, T, N) or (K*T, N)
    :y_pred: (K, T, N) or (K*T, N)
    """
    N = y.shape[-1]
    if len(y.shape) > 2:
        y = y.reshape((-1, N))
    if len(y_pred.shape) > 2:
        y_pred = y_pred.reshape((-1, N))
    r2s = np.asarray([r2_score(y[:, n].flatten(), y_pred[:, n].flatten()) for n in range(N)])
    if clip:
        return np.clip(r2s, 0., 1.)
    else:
        return r2s
        
