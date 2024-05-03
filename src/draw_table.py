from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

model = "NDT1"

mask_ratio = 0.3
mask_methods = ['mask_neuron', 'mask_causal', 'mask_intra-region', 'mask_inter-region', 'mask_temporal', 'mask_combined', 'mask_all']
eval_methods = ['co_smooth', 'forward_pred', 'intra_region', 'inter_region']
finetune_methods = ['choice_decoding', 'continuous_decoding']

if model == "NDT2":
    mask_methods += ['mask_random_token']

save_path = Path(f'/mnt/home/yzhang1/ceph/results/eval/model_{model}/method_ssl')

metrics_dict = {}
for mask in mask_methods:
    metrics_dict[mask] = {}
    if mask == 'mask_inter-region':
        ratio = 0.0
    else:
        ratio = mask_ratio
    for eval in eval_methods:
        metrics_dict[mask][eval] = {}
        try:
            r2 = np.load(save_path/mask/f'ratio_{ratio}'/eval/'r2.npy')
        except:
            r2 = np.zeros(2)
        try:
            bps = np.load(save_path/mask/f'ratio_{ratio}'/eval/'bps.npy')
        except:
            bps = 0
        metrics_dict[mask][eval]['r2_psth'] = np.nanmean(r2.T[0]) 
        metrics_dict[mask][eval]['r2_per_trial'] = np.nanmean(r2.T[1]) 
        metrics_dict[mask][eval]['bps'] = np.nanmean(bps) 
    for eval in finetune_methods:
        metrics_dict[mask][eval] = {}
        if eval == "choice_decoding":                
            try:
                acc = np.load(save_path/mask/f'ratio_{ratio}'/eval/'choice_results.npy', allow_pickle=True).item()['acc']
            except:
                acc = np.zeros(1)
            metrics_dict[mask][eval]['metric'] = acc
        elif eval == "continuous_decoding":
            try:
                r2 = np.load(
                    save_path/mask/f'ratio_{ratio}'/eval/'left-whisker-motion-energy_results.npy', allow_pickle=True
                ).item()['r2']
            except:
                r2 = np.zeros(1)
            metrics_dict[mask][eval]['metric'] = r2

N = len(mask_methods)
K = len(eval_methods)
M = len(finetune_methods)
r2_psth_mat, r2_per_trial_mat, bps_mat = np.zeros((N, K)), np.zeros((N, K)), np.zeros((N, K))
behave_mat = np.zeros((N, M))
for i, mask in enumerate(mask_methods):
    for j, eval in enumerate(eval_methods):
        r2_psth_mat[i,j] = metrics_dict[mask][eval]['r2_psth']
        r2_per_trial_mat[i,j] = metrics_dict[mask][eval]['r2_per_trial']
        bps_mat[i,j] = metrics_dict[mask][eval]['bps']
    for j, eval in enumerate(finetune_methods):
        behave_mat[i,j] = metrics_dict[mask][eval]['metric']

fig, axes = plt.subplots(1, 4, figsize=(17.5, 5))

mat = bps_mat
im0 = axes[0].imshow(mat, cmap='RdYlGn')
axes[0].set_title("co-bps")

for i in range(len(mask_methods)):
    for j in range(len(eval_methods)):
        color = 'w' if mat[i, j] < 0.5 else 'k'
        text = axes[0].text(j, i, f'{mat[i, j]:.2f}',
                       ha="center", va="center", color=color, fontsize=12)

mat = r2_psth_mat
im1 = axes[1].imshow(mat, cmap='RdYlGn')
axes[1].set_title(r"$R^2$ PSTH")

for i in range(len(mask_methods)):
    for j in range(len(eval_methods)):
        color = 'w' if mat[i, j] < 0.5 else 'k'
        text = axes[1].text(j, i, f'{mat[i, j]:.2f}',
                       ha="center", va="center", color=color, fontsize=12)

mat = r2_per_trial_mat
im2 = axes[2].imshow(mat, cmap='RdYlGn')
axes[2].set_title(r"$R^2$ per trial")

for i in range(len(mask_methods)):
    for j in range(len(eval_methods)):
        color = 'w' if mat[i, j] < 0.5 else 'k'
        text = axes[2].text(j, i, f'{mat[i, j]:.2f}',
                       ha="center", va="center", color=color, fontsize=12)

mat = behave_mat
im2 = axes[3].imshow(mat, cmap='RdYlGn')
axes[3].set_title(r"behavior decoding")

for i in range(len(mask_methods)):
    for j in range(len(finetune_methods)):
        color = 'w' if mat[i, j] < 0.5 else 'k'
        text = axes[3].text(j, i, f'{mat[i, j]:.2f}',
                       ha="center", va="center", color=color, fontsize=12)

for i, ax in enumerate(axes):
    if model == "NDT2":
        ax.set_yticks(np.arange(N), labels=['neuron mask','causal mask', 'intra-region mask', 'inter-region mask', 'temporal mask', 'neuron+temporal+causal mask', 'all mask', 'random token mask'])
    else:
        ax.set_yticks(np.arange(N), labels=['neuron mask','causal mask', 'intra-region mask', 'inter-region mask', 'temporal mask', 'neuron+temporal+causal mask', 'all mask'])
    if i < len(axes)-1:
        ax.set_xticks(np.arange(K), labels=['co-smooth','forward pred', 'intra-region', 'inter-region'])
    else:
        ax.set_xticks(np.arange(M), labels=['choice', 'whisker motion energy'])
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

fig.tight_layout()
plt.savefig(f'results/table/{model}_metrics.png')
