import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import random
import pickle
import os

#%%
class CTRNN(nn.Module):
    """Continuous-time RNN.

    Parameters:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        dt: discretization time step in ms. 
            If None, dt equals time constant tau

    Inputs:
        input: tensor of shape (seq_len, batch, input_size)
        hidden: tensor of shape (batch, hidden_size), initial hidden activity
            if None, hidden is initialized through self.init_hidden()
        
    Outputs:
        output: tensor of shape (seq_len, batch, hidden_size)
        hidden: tensor of shape (batch, hidden_size), final hidden activity
    """

    def __init__(self, input_size, N, dt=None, tau=100, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = N
        self.tau = tau
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha

        self.input2h = nn.Linear(input_size, N, bias=True) # input to region B
        
        # recurrent layer 
        self.h2h = nn.Linear(N, N, bias=False) 


    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        #print('init_hidden')
        #return torch.zeros(batch_size, self.hidden_size)
        return torch.randn(batch_size, self.hidden_size)*0.3

    def recurrence(self, input, hidden):
        """Run network for one time step.
        
        Inputs:
            input: tensor of shape (batch, input_size)
            hidden: tensor of shape (batch, hidden_size)
        
        Outputs:
            h_new: tensor of shape (batch, hidden_size),
                network activity at the next time step
        """
        #m = nn.Softplus()
        m = nn.Tanh()

        h_new = m(self.input2h(input) + self.h2h(hidden))
        hidden_new = hidden * (1 - self.alpha) + h_new * self.alpha
        
        return hidden_new

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        
        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)

        # Loop through time
        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)

        # Stack together output from all time steps
        output = torch.stack(output, dim=0)  # (seq_len, batch, hidden_size)
        return output, hidden

#%%
class RNNNet(nn.Module):
    """Recurrent network model.

    Parameters:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
    
    Inputs:
        x: tensor of shape (Seq Len, Batch, Input size)

    Outputs:
        out: tensor of shape (Seq Len, Batch, Output size)
        rnn_output: tensor of shape (Seq Len, Batch, Hidden size)
    """
    def __init__(self, input_size, N, output_size, **kwargs):
        super().__init__()

        # Continuous time RNN
        self.rnn = CTRNN(input_size, N, **kwargs)        
        # Add an output layer
        self.fc = nn.Linear(N, output_size)

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        out = self.fc(rnn_output)
        return out, rnn_output

#%%
path = '/home/ywang74/Dev/IBL_foundation_model/src/loader/'
#file_path = path + 'chaotic_rnn_500_tau25_dt10_small_g.pth'
#file_path = path + 'chaotic_rnn_500_tau25_dt10.pth'
#file_path = path + 'chaotic_rnn_600_tau25_dt10_g_5_btw_sparsity_01.pth'
file_path = path + 'chaotic_rnn_600_tau25_dt10_g_3_btw_sparsity_01.pth'

def chaotic_rnn_loader(n_trial, n1, n2, n3, T = 500, file_path = file_path):

    N = 600
    input_size = 100
    net = RNNNet(input_size, N, 2, tau=25, dt = 10)
    net.load_state_dict(torch.load(file_path))
    net.eval()

    x = torch.zeros(T, n_trial, input_size)
    
    out, activity = net(x)

    area_ind_list = np.array([0]*n1 + [1]*n2 + [2]*n3) # (n, )

    factors = 1+activity.permute(1,0,2).detach().numpy() #(n_trial, T, n_factors)

    n = n1 + n2 + n3
    spike_data = np.zeros((n_trial, T, n))
    fr = np.zeros((n_trial, T, n))

    n_factors = 200

    w1 = random(n_factors, n1, density=0.02).toarray()
    w2 = random(n_factors, n2, density=0.02).toarray()
    w3 = random(n_factors, n3, density=0.02).toarray()

    fr[:,:,:n1] = factors[:,:,:n_factors] @ w1 
    fr[:,:,n1:n1+n2] = factors[:,:,n_factors:2*n_factors] @ w2
    fr[:,:,n1+n2:] = factors[:,:,2*n_factors:3*n_factors] @ w3

    #normalized fr
    fr_norm = (fr-np.min(fr, axis=(0,1))[None,None,:])/(np.max(fr, axis=(0,1))[None,None,:]-np.min(fr, axis=(0,1))[None,None,:])*2
    fr_norm[np.isnan(fr_norm)] = 0

    spike_data = np.random.poisson(np.exp(fr_norm))

    return spike_data[:,100:,:], area_ind_list, fr_norm[:,100:,:], factors[:,100:,:]

#%%
from torch.utils.data import DataLoader, RandomSampler
class DatasetDataLoader:
    def __init__(self, datasets_list, batch_size):
        self.loaders = [DataLoader(dataset, batch_size=batch_size, sampler=RandomSampler(dataset)) for dataset in datasets_list]

    def __iter__(self):
        self.iter_loaders = [iter(loader) for loader in self.loaders]
        self.loader_order = np.random.choice(len(self.loaders), size=len(self.loaders), replace=False)
        self.ind = 0
        self.loader_exhausted = [False] * len(self.loaders)
        return self

    def __next__(self):
        if all(self.loader_exhausted):
            raise StopIteration

        loader_ind = self.loader_order[self.ind]
        self.ind = (self.ind + 1) % len(self.loader_order)  # Wrap around to the start
        try:
            return next(self.iter_loaders[loader_ind])
        except StopIteration:
            # If the DataLoader for this dataset is exhausted, mark it as exhausted
            self.loader_exhausted[loader_ind] = True
            # Try to get the next batch from the next DataLoader
            return self.__next__()

    def __len__(self):
        return sum(len(loader) for loader in self.loaders)
    
class BaseDatasetRNN(torch.utils.data.Dataset):
    def __init__(self, spike_data, choice,
                 area_ind_list, session_ind, fr, factors):
              
        self.spike_data = spike_data # K x T x N
        self.area_ind_list = area_ind_list  # list of area_ind for each neuron
        self.session_ind = session_ind

        self.behavior_data = choice[:, None, None]*np.ones((1, spike_data.shape[1], 1)) # K x T x 1
        self.fr = np.exp(fr)
        self.factors = factors
        K, T, N = self.spike_data.shape
        self.N = N
        self.T = T
        self.K = K

    def _preprocess_rnn_data(self, idx):
        # idx is the trial index
        binned_spikes_data = self.spike_data[idx].astype(np.float32) # T x N
        fr_data = self.fr[idx].astype(np.float32) # T x N
        factors_data = self.factors[idx].astype(np.float32) # T x n_factors

        time_attn_mask = np.ones((self.T,))
        space_attn_mask = np.ones((self.N,))

        spikes_timestamps = np.arange(self.T)  
        spikes_spacestamps = np.arange(self.N)

        target_behavior = self.behavior_data[idx] # T x D
        neuron_depths = np.ones((self.N,))
        neuron_regions = self.area_ind_list

        return {
                    "spikes_data": binned_spikes_data,
                    "time_attn_mask": time_attn_mask.astype(np.int64),
                    "space_attn_mask": space_attn_mask.astype(np.int64),
                    "spikes_timestamps": spikes_timestamps,
                    "spikes_spacestamps": spikes_spacestamps,
                    "target": target_behavior,
                    "neuron_depths": neuron_depths, 
                    "neuron_regions": neuron_regions,
                    "eid": self.session_ind,
                    "fr": fr_data,
                    "factors": factors_data
                }


    def __len__(self):
        return self.K
    
    def __getitem__(self, idx):
        return self._preprocess_rnn_data(idx)

#%%
def make_chaotic_rnn_loader(session_ind_list, batch_size):
    datasets = {'train': [], 'val': [], 'test': []}
    data_loader = {}
    num_neurons = []
    num_trials = {'train': [], 'val': [], 'test': []}
    area_ind_list_list = []
    record_info_list = []
    
    path = '/expanse/lustre/scratch/ywang74/temp_project/Data/synthetic/'

    generated_session_ind_list = [int(file.split('_')[-1].split('.')[0]) for file in os.listdir(path)]  

    for session_ind in session_ind_list:
        session_path = path + f'chaotic_rnn_{session_ind}.pkl'
        if session_ind in generated_session_ind_list:
            spike_data, area_ind_list, fr, factors, n1, n2, n3, K, omit_region, train_ind, val_ind, test_ind = pickle.load(open(session_path, 'rb'))
        else:
            n1 = np.random.randint(50, 60)
            n2 = np.random.randint(50, 60)
            n3 = np.random.randint(50, 60)

            K = np.random.randint(200, 300)
            #generate spike data from all 3 areas
            spike_data, area_ind_list, fr, factors = chaotic_rnn_loader(K, n1, n2, n3, T = 500)
            #randomly omit one region
            omit_region = np.random.randint(3)

            #get train/valid/test trial_ind
            indices = np.arange(K)
            np.random.shuffle(indices)
            train_ind, val_ind, test_ind = np.split(indices, [int(.6*K), int(.8*K)])
            
            pickle.dump((spike_data, area_ind_list, fr, factors, n1, n2, n3, K, omit_region, train_ind, val_ind, test_ind), open(session_path, 'wb'))

        choice = np.zeros((K,))
        #omit data from one region
        record_neuron_flag = (area_ind_list != omit_region)
        record_info = {'gt_n': [n1, n2, n3], 'omit_region': omit_region}
        record_info_list.append(record_info)
        spike_data = spike_data[:,:,record_neuron_flag==1]
        area_ind_list = area_ind_list[record_neuron_flag==1]
        
        for ind, name in zip([train_ind, val_ind, test_ind], ['train', 'val', 'test']):
            dataset = BaseDatasetRNN(spike_data[ind], choice[ind], area_ind_list, session_ind, fr[ind], factors[ind])
            num_trials[name].append(len(ind))
            datasets[name].append(dataset)

        num_neurons.append(dataset.N) # train, val, test should have the same number of neurons; so it's fine to just append the last one
        area_ind_list_list.append(area_ind_list)

    print('num_neurons: ', num_neurons)
    print('num_trials: ', num_trials)
    
    for name in ['train', 'val', 'test']:
        if batch_size < sum(num_trials[name]):
            dataset_list = datasets[name]
            if name != 'test':
                data_loader[name] =  DatasetDataLoader(dataset_list, batch_size)
            else:
                data_loader[name] =  DatasetDataLoader(dataset_list, batch_size= sum(num_trials[name])) #for test, the batch size is the total number of trials
            print('Succesfully constructing the dataloader for ', name)

    return data_loader, num_neurons, datasets, area_ind_list_list, record_info_list


# data_loader, num_neurons, _, _, _ = make_chaotic_rnn_loader([1,2], 12)

# for batch in data_loader['test']:
#     print(batch['spikes_data'].shape)
#     print(batch['target'].shape)
    
# breakpoint()
