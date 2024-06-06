from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

eid = '824cf03d-4012-4ab1-b499-c83a92c5589e'

model = "iTransformer"

ratio = 0.3
mask_methods = ['mask_neuron', 'mask_causal', 'mask_temporal', 'mask_intra-region', 'mask_inter-region', 'mask_all_prompt']
eval_methods = ['co_smooth', 'forward_pred', 'intra_region', 'inter_region']
finetune_methods = ['choice_decoding', 'continuous_decoding']

save_path = Path(f'/expanse/lustre/scratch/yzhang39/temp_project/results/{eid}/eval/model_{model}/method_ssl/')

metrics_dict = {}
for mask in mask_methods:
    metrics_dict[mask] = {}
    if mask == 'mask_all_prompt':
        fname = 'mask_all'
        prompt = 'True'
    else:
        fname = mask
        prompt = 'False'
    for eval in eval_methods:
        metrics_dict[mask][eval] = {}
        try:
            r2 = np.load(save_path/fname/f'ratio_{ratio}'/'mask_token_False'/f'prompt_{prompt}'/'NEMO_False'/'no_channel_False'/eval/'r2.npy')
        except:
            r2 = np.zeros(2)
        try:
            bps = np.load(save_path/fname/f'ratio_{ratio}'/'mask_token_False'/f'prompt_{prompt}'/'NEMO_False'/'no_channel_False'/eval/'bps.npy')
        except:
            bps = 0
        metrics_dict[mask][eval]['r2_psth'] = np.nanmean(r2.T[0]) 
        metrics_dict[mask][eval]['r2_per_trial'] = np.nanmean(r2.T[1]) 
        metrics_dict[mask][eval]['bps'] = np.nanmean(bps) 
    for eval in finetune_methods:
        metrics_dict[mask][eval] = {}
        if eval == "choice_decoding":                
            try:
                acc = np.load(save_path/fname/f'ratio_{ratio}'/'mask_token_False'/f'prompt_{prompt}'/'NEMO_False'/'no_channel_False'/eval/'choice_results.npy', allow_pickle=True).item()['acc']
            except:
                acc = np.zeros(1)
            metrics_dict[mask][eval]['metric'] = acc
        elif eval == "continuous_decoding":
            try:
                r2 = np.load(
                    save_path/fname/f'ratio_{ratio}'/'mask_token_False'/f'prompt_{prompt}'/'NEMO_False'/'no_channel_False'/eval/'whisker-motion-energy_results.npy', allow_pickle=True
                ).item()['rsquared']
            except:
                r2 = np.zeros(1)
            metrics_dict[mask][eval]['metric'] = r2


N = len(mask_methods)
K = len(eval_methods)
M = len(finetune_methods) 
P = 0
r2_psth_mat, r2_per_trial_mat, bps_mat = np.zeros((N, K)), np.zeros((N, K)), np.zeros((N, K))
behave_mat = np.zeros((N + P, M))
for i, mask in enumerate(mask_methods):
    for j, eval in enumerate(eval_methods):
        r2_psth_mat[i,j] = metrics_dict[mask][eval]['r2_psth']
        r2_per_trial_mat[i,j] = metrics_dict[mask][eval]['r2_per_trial']
        bps_mat[i,j] = metrics_dict[mask][eval]['bps']
    for j, eval in enumerate(finetune_methods):
        behave_mat[i,j] = metrics_dict[mask][eval]['metric']


def rect(pos, ax):
    r = plt.Rectangle(pos-0.5, 1,1, facecolor="none", edgecolor="k", linewidth=.5)
    ax.add_patch(r)

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

mat = bps_mat
im0 = axes[0].imshow(mat, cmap=ListedColormap(['white']))
axes[0].set_title("co-bps")

x,y = np.meshgrid(np.arange(mat.shape[1]),np.arange(mat.shape[0]))
m = np.c_[x[mat.astype(bool)],y[mat.astype(bool)]]
for pos in m:
    rect(pos, axes[0])

for j in range(len(eval_methods)):
    for i in range(len(mask_methods)):
        color = 'k'
        if mat[i, j] == mat[:, j].max(): 
            text = axes[0].text(j, i, f'{mat[i, j]:.2f}',
                       ha="center", va="center", color=color, fontsize=11, weight='bold')
        else:
            text = axes[0].text(j, i, f'{mat[i, j]:.2f}',
                       ha="center", va="center", color=color, fontsize=11)

mat = r2_psth_mat
im1 = axes[1].imshow(mat, cmap=ListedColormap(['white']))
axes[1].set_title(r"$R^2$ PSTH")

x,y = np.meshgrid(np.arange(mat.shape[1]),np.arange(mat.shape[0]))
m = np.c_[x[mat.astype(bool)],y[mat.astype(bool)]]
for pos in m:
    rect(pos, axes[1])

for i in range(len(mask_methods)):
    for j in range(len(eval_methods)):
        color = 'k'
        if mat[i, j] == mat[:, j].max(): 
            text = axes[1].text(j, i, f'{mat[i, j]:.2f}',
                       ha="center", va="center", color=color, fontsize=11, weight='bold')
        else:
            text = axes[1].text(j, i, f'{mat[i, j]:.2f}',
                       ha="center", va="center", color=color, fontsize=11)

mat = r2_per_trial_mat
im2 = axes[2].imshow(mat, cmap=ListedColormap(['white']))
axes[2].set_title(r"$R^2$ per trial")

x,y = np.meshgrid(np.arange(mat.shape[1]),np.arange(mat.shape[0]))
m = np.c_[x[mat.astype(bool)],y[mat.astype(bool)]]
for pos in m:
    rect(pos, axes[2])

for i in range(len(mask_methods)):
    for j in range(len(eval_methods)):
        color = 'k'
        if mat[i, j] == mat[:, j].max(): 
            text = axes[2].text(j, i, f'{mat[i, j]:.2f}',
                       ha="center", va="center", color=color, fontsize=11, weight='bold')
        else:
            text = axes[2].text(j, i, f'{mat[i, j]:.2f}',
                       ha="center", va="center", color=color, fontsize=11)

mat = behave_mat
im2 = axes[3].imshow(mat, cmap=ListedColormap(['white']))
axes[3].set_title(r"behavior decoding")

x,y = np.meshgrid(np.arange(mat.shape[1]),np.arange(mat.shape[0]))
m = np.c_[x[mat.astype(bool)],y[mat.astype(bool)]]
for pos in m:
    rect(pos, axes[3])

for i in range(N+P):
    for j in range(len(finetune_methods)):
        color = 'k'
        if mat[i, j] == mat[:, j].max(): 
            text = axes[3].text(j, i, f'{mat[i, j]:.2f}',
                       ha="center", va="center", color=color, fontsize=11, weight='bold')
        else:
            text = axes[3].text(j, i, f'{mat[i, j]:.2f}',
                       ha="center", va="center", color=color, fontsize=11)

for i, ax in enumerate(axes):
    if model == "NDT2":
        if i == 0:
            ax.set_yticks(np.arange(N), labels=['neuron mask','causal mask', 'intra-region mask', 'inter-region mask', 'temporal mask', 'neuron+temporal+causal mask', 'all mask', 'random token mask'])
        else:
            ax.set_yticks([],[])
    else:
        if i == 0:
            ax.set_yticks(np.arange(N), labels=['neuron mask','causal mask', 'temporal mask', 'intra-region mask', 'inter-region mask', 'all mask + prompt'])
        else:
            ax.set_yticks([],[])
    if i < len(axes)-1:
        ax.set_xticks(np.arange(K), labels=['co-smooth','forward pred', 'intra-region', 'inter-region'])
    else:
        ax.set_xticks(np.arange(M), labels=['choice', 'whisker motion energy'])
        if i == 0:
            ax.set_yticks(np.arange(N+P), 
                    labels=['neuron mask','causal mask', 'temporal mask', 'intra-region mask', 'inter-region mask', 'all mask + prompt'])
        else:
            ax.set_yticks([],[])
            
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    

# fig.tight_layout()
plt.savefig(f'results/table/{model}_metrics.png')
