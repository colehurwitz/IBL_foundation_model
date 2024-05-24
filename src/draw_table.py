from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

model = "NDT2"

mask_methods = ['mask_neuron', 'mask_causal', 'mask_intra-region', 'mask_inter-region', 'mask_temporal']
eval_methods = ['co_smooth', 'forward_pred', 'intra_region', 'inter_region', 'choice_decoding', 'continuous_decoding']

if model == "NDT2":
    mask_methods += ['mask_random_token']

save_path = Path(f'/home/yizi/IBL_foundation_model/figs/model_{model}/method_ssl')

metrics_dict = {}
for mask in mask_methods:
    metrics_dict[mask] = {}
    for eval in eval_methods:
        metrics_dict[mask][eval] = {}
        if eval == "choice_decoding":
            try:
                acc = np.load(save_path/mask/eval/'choice_results.npy', allow_pickle=True).item()['acc']
            except:
                acc = np.zeros(1)
            metrics_dict[mask][eval]['r2_psth'] = acc
            metrics_dict[mask][eval]['r2_per_trial'] = acc
            metrics_dict[mask][eval]['bps'] = acc
        elif eval == "continuous_decoding":
            try:
                r2 = np.load(
                    save_path/mask/eval/'left-whisker-motion-energy_results.npy', allow_pickle=True
                ).item()['r2']
            except:
                r2 = np.zeros(1)
            metrics_dict[mask][eval]['r2_psth'] = r2
            metrics_dict[mask][eval]['r2_per_trial'] = r2
            metrics_dict[mask][eval]['bps'] = r2
        else:
            try:
                r2 = np.load(save_path/mask/eval/'r2.npy')
            except:
                r2 = np.zeros(2)
            try:
                bps = np.load(save_path/mask/eval/'bps.npy')
            except:
                bps = 0
            metrics_dict[mask][eval]['r2_psth'] = np.nanmean(r2.T[0]) if np.nanmean(r2.T[0]) > -10 else -5
            metrics_dict[mask][eval]['r2_per_trial'] = np.nanmean(r2.T[1]) if np.nanmean(r2.T[1]) > -10 else -5
            metrics_dict[mask][eval]['bps'] = np.nanmean(bps) if np.nanmean(bps) > -10 else -5

N = len(mask_methods)
K = len(eval_methods)
r2_psth_mat, r2_per_trial_mat, bps_mat = np.zeros((N, K)), np.zeros((N, K)), np.zeros((N, K))
for i, mask in enumerate(mask_methods):
    for j, eval in enumerate(eval_methods):
        r2_psth_mat[i,j] = metrics_dict[mask][eval]['r2_psth']
        r2_per_trial_mat[i,j] = metrics_dict[mask][eval]['r2_per_trial']
        bps_mat[i,j] = metrics_dict[mask][eval]['bps']

fig, axes = plt.subplots(1,3, figsize=(12,4))

mat = bps_mat
im0 = axes[0].imshow(mat, cmap='RdYlGn')
axes[0].set_title("bps")

for i in range(len(mask_methods)):
    for j in range(len(eval_methods)):
        color = 'w' if mat[i, j] < 0.5 else 'k'
        text = axes[0].text(j, i, f'{mat[i, j]:.2f}',
                       ha="center", va="center", color=color, fontsize=12)

mat = r2_psth_mat
im1 = axes[1].imshow(mat, cmap='RdYlGn')
axes[1].set_title("r2 psth")

for i in range(len(mask_methods)):
    for j in range(len(eval_methods)):
        color = 'w' if mat[i, j] < 0.5 else 'k'
        text = axes[1].text(j, i, f'{mat[i, j]:.2f}',
                       ha="center", va="center", color=color, fontsize=12)

mat = r2_per_trial_mat
im2 = axes[2].imshow(mat, cmap='RdYlGn')
axes[2].set_title("r2 per trial")

for i in range(len(mask_methods)):
    for j in range(len(eval_methods)):
        color = 'w' if mat[i, j] < 0.5 else 'k'
        text = axes[2].text(j, i, f'{mat[i, j]:.2f}',
                       ha="center", va="center", color=color, fontsize=12)

for ax in axes:
    if model == "NDT2":
        ax.set_yticks(np.arange(N), labels=['neuron mask','causal mask', 'intra-region mask', 'inter-region mask', 'temporal mask', 'random token mask'])
    else:
        ax.set_yticks(np.arange(N), labels=['neuron mask','causal mask', 'intra-region mask', 'inter-region mask', 'temporal mask'])
    ax.set_xticks(np.arange(K), labels=['co-smooth','forward pred', 'intra-region', 'inter-region', 'choice decoding', 'continuous decoding'])
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

fig.tight_layout()
plt.savefig(f'figs/table/{model}_metrics.png')
