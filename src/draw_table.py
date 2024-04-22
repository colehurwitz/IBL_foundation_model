from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

mask_methods = ['mask_neuron', 'mask_temporal', 'mask_intra-region', 'mask_inter-region']
eval_methods = ['co_smooth', 'forward_pred', 'intra_region', 'inter_region']

save_path = Path('/home/yizi/IBL_foundation_model/figs')

metrics_dict = {}
for mask in mask_methods:
    metrics_dict[mask] = {}
    for eval in eval_methods:
        metrics_dict[mask][eval] = {}
        try:
            r2 = np.load(save_path/mask/eval/'r2.npy')
        except:
            r2 = np.zeros(2)
        try:
            bps = np.load(save_path/mask/eval/'bps.npy')
        except:
            bps = 0
        metrics_dict[mask][eval]['r2_psth'] = np.nanmean(r2.T[0]) if np.nanmean(r2.T[0]) > -10 else -1
        metrics_dict[mask][eval]['r2_per_trial'] = np.nanmean(r2.T[1]) if np.nanmean(r2.T[1]) > -10 else -1
        metrics_dict[mask][eval]['bps'] = np.nanmean(bps) if np.nanmean(bps) > -10 else -1

r2_psth_mat, r2_per_trial_mat, bps_mat = np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4))
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
    ax.set_yticks(np.arange(4), labels=['neuron mask','causal mask', 'intra-region mask', 'inter-region mask'])
    ax.set_xticks(np.arange(4), labels=['co-smooth','forward pred', 'intra-region', 'inter-region'])
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

fig.tight_layout()
plt.savefig('results/table/metrics.png')