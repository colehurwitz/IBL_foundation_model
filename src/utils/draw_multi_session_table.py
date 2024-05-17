from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='NDT1')
parser.add_argument('--base_path', type=str, default='/mnt/home/yzhang1/ceph/results/eval')
parser.add_argument('--num_train_sessions', type=int, default=10)
args = parser.parse_args()

model = args.model
num_train_sessions = args.num_train_sessions

if model == "NDT1":
    stitch = 'True'
else:
    stitch = 'False'
ratio = 0.3
mask_methods = ['mask_temporal', 'mask_all_prompt']
eval_methods = ['co_smooth', 'forward_pred', 'intra_region', 'inter_region']
finetune_methods = ['choice_decoding', 'continuous_decoding']
behavior_decoders = ['linear', 'reduced-rank', 'mlp']
behavior_decoders = []

if model == "NDT2":
    mask_methods += ['mask_random_token']

save_path = os.path.join(args.base_path, 
                         'results', 
                         'eval',
                         f'num_session_{num_train_sessions}',
                         f'model_{model}', 
                         'method_ssl')
# get eids under save_path
def get_test_re_eids(eids_path):
    with open(eids_path) as file:
        include_eids = [line.rstrip() for line in file]
    return include_eids

eids = get_test_re_eids(os.path.join(args.base_path, 'data','test_re_eids.txt'))
metrics_dict = {}
for eid in eids:
    metrics_dict[eid] = {}
    for mask in mask_methods:
        metrics_dict[eid][mask] = {}
        if mask == 'mask_all_prompt':
            fname = 'mask_all'
            prompt = 'True'
        else:
            fname = mask
            prompt = 'False'
        for eval in eval_methods:
            metrics_dict[eid][mask][eval] = {}
            try:
                r2_path = os.path.join(save_path, fname, f'stitch_{stitch}', eid, eval, 'r2.npy')
                r2 = np.load(r2_path)
                print(f'{eval} success')
            except:
                r2 = np.zeros(2)
            try:
                bps_path = os.path.join(save_path, fname, f'stitch_{stitch}', eid, eval, 'bps.npy')
                bps = np.load(bps_path)
                print(f'{eval} success')
            except:
                bps = 0
            metrics_dict[eid][mask][eval]['r2_psth'] = np.nanmean(r2.T[0]) 
            metrics_dict[eid][mask][eval]['r2_per_trial'] = np.nanmean(r2.T[1]) 
            metrics_dict[eid][mask][eval]['bps'] = np.nanmean(bps) 
        for eval in finetune_methods:
            metrics_dict[eid][mask][eval] = {}
            if eval == "choice_decoding":                
                try:
                    acc_path = os.path.join(save_path, fname, f'stitch_{stitch}', eid, eval, 'choice_results.npy')
                    acc = np.load(acc_path, allow_pickle=True).item()['acc']
                    print('choice_decoding success')
                except:
                    acc = np.zeros(1)
                metrics_dict[eid][mask][eval]['metric'] = acc
            elif eval == "continuous_decoding":
                try:
                    r2_path = os.path.join(save_path, fname, f'stitch_{stitch}', eid, eval, 'whisker-motion-energy_results.npy')
                    r2 = np.load(
                        r2_path, allow_pickle=True
                    ).item()['rsquared']
                    print('continuous_decoding success')
                except:
                    r2 = np.zeros(1)
                metrics_dict[eid][mask][eval]['metric'] = r2


    decode_metrics = {}
    for decoder in behavior_decoders:
        decode_metrics[decoder] = {}
        for eval in finetune_methods:
            decode_metrics[decoder][eval] = {}
            if eval == "choice_decoding":
                try:
                    acc = np.load(f'/mnt/home/yzhang1/ceph/results/decoding/choice/{decoder}/671c7ea7-6726-4fbe-adeb-f89c2c8e489b.npy', allow_pickle=True).item()['test_metric']
                except:
                    acc = np.zeros(1)
                decode_metrics[decoder][eval][eval] = acc
            elif eval == "continuous_decoding":
                try:
                    r2 = np.load(f'/mnt/home/yzhang1/ceph/results/decoding/left-whisker-motion-energy/{decoder}/671c7ea7-6726-4fbe-adeb-f89c2c8e489b.npy', allow_pickle=True).item()['test_metric']
                except:
                    r2 = np.zeros(1)
                decode_metrics[decoder][eval][eval] = r2



    N = len(mask_methods)
    K = len(eval_methods)
    M = len(finetune_methods) 
    P = len(behavior_decoders)
    r2_psth_mat, r2_per_trial_mat, bps_mat = np.zeros((N, K)), np.zeros((N, K)), np.zeros((N, K))
    behave_mat = np.zeros((N + P, M))
    for i, mask in enumerate(mask_methods):
        for j, eval in enumerate(eval_methods):
            r2_psth_mat[i,j] = metrics_dict[eid][mask][eval]['r2_psth']
            r2_per_trial_mat[i,j] = metrics_dict[eid][mask][eval]['r2_per_trial']
            bps_mat[i,j] = metrics_dict[eid][mask][eval]['bps']
        for j, eval in enumerate(finetune_methods):
            behave_mat[i,j] = metrics_dict[eid][mask][eval]['metric']
    
    for i, decoder in enumerate(behavior_decoders):
        for j, eval in enumerate(finetune_methods):
            behave_mat[i+N,j] = decode_metrics[decoder][eval]['metric']


    fig, axes = plt.subplots(1, 4, figsize=(20, 7))
    # set the title
    fig.suptitle(f"model: {model}, eid: {eid}")
    mat = bps_mat
    im0 = axes[0].imshow(mat, cmap='Blues_r')
    axes[0].set_title("co-bps")

    for j in range(len(eval_methods)):
        for i in range(len(mask_methods)):
            color = 'w' if mat[i, j] < -0.5 else 'k'
            if mat[i, j] == mat[:, j].max(): 
                text = axes[0].text(j, i, f'{mat[i, j]:.2f}',
                        ha="center", va="center", color=color, fontsize=12, weight='bold')
            else:
                text = axes[0].text(j, i, f'{mat[i, j]:.2f}',
                        ha="center", va="center", color=color, fontsize=12)

    mat = r2_psth_mat
    im1 = axes[1].imshow(mat, cmap='Blues_r')
    axes[1].set_title(r"$R^2$ PSTH")

    for i in range(len(mask_methods)):
        for j in range(len(eval_methods)):
            color = 'w' if mat[i, j] < -0.5 else 'k'
            if mat[i, j] == mat[:, j].max(): 
                text = axes[1].text(j, i, f'{mat[i, j]:.2f}',
                        ha="center", va="center", color=color, fontsize=12, weight='bold')
            else:
                text = axes[1].text(j, i, f'{mat[i, j]:.2f}',
                        ha="center", va="center", color=color, fontsize=12)

    mat = r2_per_trial_mat
    im2 = axes[2].imshow(mat, cmap='Blues_r')
    axes[2].set_title(r"$R^2$ per trial")

    for i in range(len(mask_methods)):
        for j in range(len(eval_methods)):
            color = 'w' if mat[i, j] < -0.5 else 'k'
            if mat[i, j] == mat[:, j].max(): 
                text = axes[2].text(j, i, f'{mat[i, j]:.2f}',
                        ha="center", va="center", color=color, fontsize=12, weight='bold')
            else:
                text = axes[2].text(j, i, f'{mat[i, j]:.2f}',
                        ha="center", va="center", color=color, fontsize=12)

    mat = behave_mat
    im2 = axes[3].imshow(mat, cmap='Blues_r')
    axes[3].set_title(r"behavior decoding")

    for i in range(N+P):
        for j in range(len(finetune_methods)):
            color = 'w' if mat[i, j] < -0.5 else 'k'
            if mat[i, j] == mat[:, j].max(): 
                text = axes[3].text(j, i, f'{mat[i, j]:.2f}',
                        ha="center", va="center", color=color, fontsize=12, weight='bold')
            else:
                text = axes[3].text(j, i, f'{mat[i, j]:.2f}',
                        ha="center", va="center", color=color, fontsize=12)

    for i, ax in enumerate(axes):
        if model == "NDT2":
            ax.set_yticks(np.arange(N), labels=['neuron mask','causal mask', 'intra-region mask', 'inter-region mask', 'temporal mask', 'neuron+temporal+causal mask', 'all mask', 'random token mask'])
        else:
            ax.set_yticks(np.arange(N), labels=['temporal mask', 'all mask + prompt'])
        if i < len(axes)-1:
            ax.set_xticks(np.arange(K), labels=['co-smooth','forward pred', 'intra-region', 'inter-region'])
        else:
            ax.set_xticks(np.arange(M), labels=['choice', 'whisker motion energy'])
            ax.set_yticks(np.arange(N+P), 
                    labels=['temporal mask', 'all mask + prompt'])
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

    fig.tight_layout()
    output_dir = os.path.join(args.base_path, 
                            'results', 
                            'table',
                            'num_session_{}'.format(num_train_sessions),
                            'model_{}'.format(model),
                            eid
                            )
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/metrics.png')
    plt.close()

print('done')

# average across eids
avg_metrics_dict = {}

for eid in eids:
    for mask in mask_methods:
        for eval in eval_methods:
            for metric in ['r2_psth', 'r2_per_trial', 'bps']:
                if mask not in avg_metrics_dict:
                    avg_metrics_dict[mask] = {}
                if eval not in avg_metrics_dict[mask]:
                    avg_metrics_dict[mask][eval] = {}
                if metric not in avg_metrics_dict[mask][eval]:
                    avg_metrics_dict[mask][eval][metric] = []
                avg_metrics_dict[mask][eval][metric].append(metrics_dict[eid][mask][eval][metric])
        for eval in finetune_methods:
            if eval not in avg_metrics_dict[mask]:
                avg_metrics_dict[mask][eval] = {}
            if 'metric' not in avg_metrics_dict[mask][eval]:
                avg_metrics_dict[mask][eval]['metric'] = []
            avg_metrics_dict[mask][eval]['metric'].append(metrics_dict[eid][mask][eval]['metric'])

N = len(mask_methods)
K = len(eval_methods)
M = len(finetune_methods) 
P = len(behavior_decoders)
r2_psth_mat, r2_per_trial_mat, bps_mat = np.zeros((N, K)), np.zeros((N, K)), np.zeros((N, K))
behave_mat = np.zeros((N + P, M))
for i, mask in enumerate(mask_methods):
    for j, eval in enumerate(eval_methods):
        r2_psth_mat[i,j] = np.mean(avg_metrics_dict[mask][eval]['r2_psth'])
        r2_per_trial_mat[i,j] = np.mean(avg_metrics_dict[mask][eval]['r2_per_trial'])
        bps_mat[i,j] = np.mean(avg_metrics_dict[mask][eval]['bps'])
    for j, eval in enumerate(finetune_methods):
        _sum_temp =sum(avg_metrics_dict[mask][eval]['metric'])
        behave_mat[i,j] = _sum_temp/len(avg_metrics_dict[mask][eval]['metric'])

fig, axes = plt.subplots(1, 4, figsize=(20, 7))
# set the title
fig.suptitle(f"model: {model}, eid: {eid}")
mat = bps_mat
im0 = axes[0].imshow(mat, cmap='Blues_r')
axes[0].set_title("co-bps")

for j in range(len(eval_methods)):
    for i in range(len(mask_methods)):
        color = 'w' if mat[i, j] < -0.5 else 'k'
        if mat[i, j] == mat[:, j].max(): 
            text = axes[0].text(j, i, f'{mat[i, j]:.2f}',
                    ha="center", va="center", color=color, fontsize=12, weight='bold')
        else:
            text = axes[0].text(j, i, f'{mat[i, j]:.2f}',
                    ha="center", va="center", color=color, fontsize=12)

mat = r2_psth_mat
im1 = axes[1].imshow(mat, cmap='Blues_r')
axes[1].set_title(r"$R^2$ PSTH")

for i in range(len(mask_methods)):
    for j in range(len(eval_methods)):
        color = 'w' if mat[i, j] < -0.5 else 'k'
        if mat[i, j] == mat[:, j].max(): 
            text = axes[1].text(j, i, f'{mat[i, j]:.2f}',
                    ha="center", va="center", color=color, fontsize=12, weight='bold')
        else:
            text = axes[1].text(j, i, f'{mat[i, j]:.2f}',
                    ha="center", va="center", color=color, fontsize=12)

mat = r2_per_trial_mat
im2 = axes[2].imshow(mat, cmap='Blues_r')
axes[2].set_title(r"$R^2$ per trial")

for i in range(len(mask_methods)):
    for j in range(len(eval_methods)):
        color = 'w' if mat[i, j] < -0.5 else 'k'
        if mat[i, j] == mat[:, j].max(): 
            text = axes[2].text(j, i, f'{mat[i, j]:.2f}',
                    ha="center", va="center", color=color, fontsize=12, weight='bold')
        else:
            text = axes[2].text(j, i, f'{mat[i, j]:.2f}',
                    ha="center", va="center", color=color, fontsize=12)

mat = behave_mat
im2 = axes[3].imshow(mat, cmap='Blues_r')
axes[3].set_title(r"behavior decoding")

for i in range(N+P):
    for j in range(len(finetune_methods)):
        color = 'w' if mat[i, j] < -0.5 else 'k'
        if mat[i, j] == mat[:, j].max(): 
            text = axes[3].text(j, i, f'{mat[i, j]:.2f}',
                    ha="center", va="center", color=color, fontsize=12, weight='bold')
        else:
            text = axes[3].text(j, i, f'{mat[i, j]:.2f}',
                    ha="center", va="center", color=color, fontsize=12)

for i, ax in enumerate(axes):
    if model == "NDT2":
        ax.set_yticks(np.arange(N), labels=['neuron mask','causal mask', 'intra-region mask', 'inter-region mask', 'temporal mask', 'neuron+temporal+causal mask', 'all mask', 'random token mask'])
    else:
        ax.set_yticks(np.arange(N), labels=['temporal mask', 'all mask + prompt'])
    if i < len(axes)-1:
        ax.set_xticks(np.arange(K), labels=['co-smooth','forward pred', 'intra-region', 'inter-region'])
    else:
        ax.set_xticks(np.arange(M), labels=['choice', 'whisker motion energy'])
        ax.set_yticks(np.arange(N+P), 
                labels=['temporal mask', 'all mask + prompt'])
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

fig.tight_layout()
output_dir = os.path.join(args.base_path, 
                        'results', 
                        'table',
                        'num_session_{}'.format(num_train_sessions),
                        'model_{}'.format(model),
                        )
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f'{output_dir}/avg_eid_metrics.png')
plt.close()

print('done')