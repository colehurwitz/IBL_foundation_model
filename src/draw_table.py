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

TASK2NAME = {'co_smooth': 'Co-Smooth', 'forward_pred':'Forward Prediction', 'intra_region':'Intra-Region', 'inter_region':'Inter-Region','choice_decoding':'Choice', 'continuous_decoding':'Whisker Motion Energy'}

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
            except:
                print(f'Failed to load: {eid}, {mask}, {eval}')
                r2 = np.zeros(2)
            try:
                bps_path = os.path.join(save_path, fname, f'stitch_{stitch}', eid, eval, 'bps.npy')
                bps = np.load(bps_path)
            except:
                print(f'Failed to load: {eid}, {mask}, {eval}')
                bps = 0
            metrics_dict[eid][mask][eval]['r2_psth'] = np.nanmean(r2.T[0]) 
            metrics_dict[eid][mask][eval]['r2_per_trial'] = np.nanmean(r2.T[1]) 
            metrics_dict[eid][mask][eval]['bps'] = np.nanmean(bps) 
        for eval in finetune_methods:
            metrics_dict[eid][mask][eval] = {}
            if eval == "choice_decoding":                
                try:
                    acc_path = os.path.join(save_path, fname, f'stitch_{stitch}', eid, eval, 'choice_results.npy')
                    _temp = np.load(acc_path, allow_pickle=True).item()
                    acc_list = {}
                    for prompt_mode in _temp:
                        _acc = _temp[prompt_mode]['acc']
                        acc_list[prompt_mode] = _acc
                    acc = acc_list['choice_neuron']
                    # acc = np.mean(list(acc_list.values()))
                    print(f'choice decoding, {eid}, {mask}: {acc_list}')
                except:
                    print(f'Failed to load: {eid}, {mask}, {eval}')
                    acc = np.zeros(1)
                metrics_dict[eid][mask][eval]['metric'] = acc
            elif eval == "continuous_decoding":
                try:
                    r2_path = os.path.join(save_path, fname, f'stitch_{stitch}', eid, eval, 'whisker-motion-energy_results.npy')
                    r2_list = {}
                    _temp = np.load(r2_path, allow_pickle=True).item()
                    for prompt_mode in _temp:
                        _r2 = _temp[prompt_mode]['rsquared']
                        r2_list[prompt_mode] = _r2
                    r2 = r2_list['whisker-motion-energy_causal']
                    # r2 = np.mean(list(r2_list.values()))
                    print(f'continuous decoding, {eid}, {mask}: {r2_list}')
                except:
                    print(f'Failed to load: {eid}, {mask}, {eval}')
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
fig.suptitle(f"model: {model}, Average across eids")
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
        ax.set_yticks(np.arange(N), labels=['Temporal Mask', 'All mask + Prompt'])
    if i < len(axes)-1:
        ax.set_xticks(np.arange(K), labels=['Co-Smooth','Forward Prediction', 'Intra-Region', 'Inter-Region'])
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


plt.rc("figure", dpi=100)
SMALL_SIZE = 10
BIGGER_SIZE = 20
plt.rc('font', size=BIGGER_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=BIGGER_SIZE)
plt.rc('axes', linewidth=1)
plt.rc('xtick', labelsize=BIGGER_SIZE)
plt.rc('ytick', labelsize=BIGGER_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=3)
# scatter plot of different eids
fig = plt.figure(figsize=(24, 4))
# plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.15, wspace=0.3, hspace=0.4)

nrows = 1
ncols = 6

xlim = [[-.1, 1.3], [-.2, .5], [-1, 1], [-.3, 1], [.5, 1], [-1, 1]]
ylim = [[0, 2], [0, 1], [0, 2], [0, 2], [0.4, 1], [-1, 1]]

if model == 'NDT1':
    temp_name = 'Temporal'
else:
    temp_name = 'random_token'
linear_line = np.linspace(-10, 10, 1000)

for i, task in enumerate(eval_methods):
    ax = fig.add_subplot(nrows, ncols, i+1)
    x_list, y_list = [], []
    for eid in eids:
        x = metrics_dict[eid]['mask_temporal'][task]['bps']
        y = metrics_dict[eid]['mask_all_prompt'][task]['bps']
        ax.scatter(x, y,s=50,c='dodgerblue')
        x_list.append(x)
        y_list.append(y)
    ax.plot(linear_line,linear_line, c='k', ls='--', lw=2)
    # adjust xlim and ylim
    xlim[i] = [min(x_list) - 0.1, max(x_list) + 0.1]
    ylim[i] = [min(y_list) - 0.1, max(y_list) + 0.1]
    if task == 'intra_region':
        ylim[i] = [min(y_list) - 0.3, max(y_list) + 0.1]
    ax.set_xlim(xlim[i])
    ax.set_ylim(ylim[i])
    ax.set_xlabel(f'{temp_name} Mask')
    if i == 0:
        ax.set_ylabel('MtM (Prompt)')
    # ax.set_ylabel('All mask + Prompt')
    ax.set_title(TASK2NAME[task])
    # set grid
    ax.grid()
    ax.set_aspect('equal', adjustable='datalim')
    # ax.legend()

for i, task in enumerate(finetune_methods):
    ax = fig.add_subplot(nrows, ncols, i+1+len(eval_methods))
    x_list, y_list = [], []
    for eid in eids:
        x = metrics_dict[eid]['mask_temporal'][task]['metric']
        y = metrics_dict[eid]['mask_all_prompt'][task]['metric']
        ax.scatter(x, y,s=50,c='dodgerblue')
        x_list.append(x)
        y_list.append(y)
    ax.plot(linear_line,linear_line, c='k', ls='--', lw=2)
    xlim[i+len(eval_methods)] = [min(x_list) - 0.1, max(x_list) + 0.1]
    ylim[i+len(eval_methods)] = [min(y_list) - 0.1, max(y_list) + 0.1]
    ax.set_xlim(xlim[i+len(eval_methods)])
    ax.set_ylim(ylim[i+len(eval_methods)])
    ax.set_xlabel(f'{temp_name} mask')
    # ax.set_ylabel('all mask + prompt')
    ax.set_title(TASK2NAME[task])
    ax.grid()
    ax.set_aspect('equal', adjustable='datalim')
    # ax.legend()
plt.tight_layout(pad=0.70)
# plt.tight_layout()
# plt.savefig(f'{output_dir}/scatter.eps', format='eps')
plt.savefig(f'{output_dir}/scatter.png')