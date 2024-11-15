import torch
import torch.nn as nn
#import torch.nn.functional as F
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

class CTRNN_3regions(nn.Module):
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

    def __init__(self, input_size, A_size, B_size, C_size, dt=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.A_size = A_size
        self.B_size = B_size
        self.C_size = C_size
        self.hidden_size = A_size + B_size + C_size
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha

        self.input2B = nn.Linear(input_size, B_size, bias=True) # input to region B
        
        # recurrent layer between region A,B,C
        self.A2A = nn.Linear(A_size, A_size, bias=False) 
        self.B2B = nn.Linear(B_size, B_size, bias=False)
        self.C2C = nn.Linear(C_size, C_size, bias=False)

        self.B2A = nn.Linear(B_size, A_size, bias=False) # region B to region A
        self.C2B = nn.Linear(C_size, B_size, bias=False) # region C to region B
        self.A2C = nn.Linear(A_size, C_size, bias=False) # region A to region C
        self.A2B = nn.Linear(A_size, B_size, bias=False) # region A to region B

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        #print('init_hidden')
        return torch.zeros(batch_size, self.hidden_size)

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

        #h_new = m(self.input2h(input) + self.h2h(hidden))
        A = hidden[:, :self.A_size]
        B = hidden[:, self.A_size:self.A_size+self.B_size]
        C = hidden[:, self.A_size+self.B_size:]

        A_h_new = m(self.A2A(A) + self.B2A(B))
        B_h_new = m(self.B2B(B) + self.C2B(C) + self.input2B(input) + self.A2B(A))
        C_h_new = m(self.C2C(C) + self.A2C(A))

        h_new = torch.cat((A_h_new, B_h_new, C_h_new), dim=1) # total input f(wr+input)

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


class RNNNet_3region(nn.Module):
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
    def __init__(self, input_size, A_size, B_size, C_size, output_size, **kwargs):
        super().__init__()

        # Continuous time RNN
        self.rnn = CTRNN_3regions(input_size, A_size, B_size, C_size, **kwargs)
        
        # Add an output layer
        self.fc_C = nn.Linear(C_size, output_size)
        self.C_size = C_size
        self.A_size = A_size
        self.B_size = B_size
        self.fc_A = nn.Linear(A_size, output_size)

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        out_C = self.fc_C(rnn_output[:, :,self.A_size+self.B_size:])
        out_A = self.fc_A(rnn_output[:, :, :self.A_size])

        return out_A, out_C, rnn_output

#%%
def auditory_WM(batch_size = 10, spont = 5, sample = 5, delay = 5, go=1, seq_len =100, std = 0.5):
    """Auditory working memory task.
    
    Parameters:
        std: standard deviation of noise
        
    Returns:
        input: tensor of shape (seq_len, batch, input_size)
        target: tensor of shape (seq_len, batch, output_size)
    """
    input_size = 3
    output_size = 2

    input = torch.zeros(seq_len, batch_size, input_size)

    # auditory stim
    tone = torch.randint(2, (batch_size,)) # 0 or 1

    # sample period
    input[spont:spont+sample, tone==0, 0] = 1
    input[spont:spont+sample, tone==1, 1] = 1
    input[spont+sample+delay: go+spont+sample+delay, :, 2] = 1 # go cue

    # add noise
    input += torch.randn(seq_len, batch_size, input_size) * std

    target = torch.zeros(seq_len, batch_size, output_size)
    target2 = torch.zeros(seq_len, batch_size, output_size)

    target[spont+sample+delay+go:, tone==0, 0] = 1 
    target[spont+sample+delay+go:, tone==1, 1] = 1 

    target2[spont+sample+delay+go:, tone==0, 0] = 1 + torch.sin(2*torch.linspace(0, 2*np.pi, seq_len)[spont+sample+delay+go:])[:,None]
    target2[spont+sample+delay+go:, tone==1, 1] = 1 + torch.sin(2*torch.linspace(0, 2*np.pi, seq_len)[spont+sample+delay+go:])[:,None]

    return input, target, target2

#%%
path = '/mnt/smb/locker/miller-locker/users/jx2484/code/autoencoder/src/loader/'
file_path = path + 'rnn_model_2_output_tanh_a2_b3_c3.pth'

def rnn_data_loader(n_trial, n1, n2, n3, T = 120, file_path = file_path):

    rnn = RNNNet_3region(input_size=3, A_size=2, B_size=3, C_size=3, output_size=2, dt=10)
    rnn.load_state_dict(torch.load(file_path))
    rnn.eval()

    inputs, labels, target2 = auditory_WM(batch_size=n_trial, spont=20, sample = 20, delay=30, go=10, seq_len=T, std=0.5)
    outputA, outputC, activity = rnn(inputs)
    
    stim = labels[-1,:,:].argmax(dim=1) # (n_trial, )
    choice = outputA[-1,:,:].argmax(dim=1) # (n_trial, )
    area_ind_list = np.array([0]*n1 + [1]*n2 + [2]*n3) # (n, )

    factors = 1+activity.permute(1,0,2).detach().numpy() #(n_trial, T, n_factors)

    n = n1 + n2 + n3
    spike_data = np.zeros((n_trial, T, n))
    fr = np.zeros((n_trial, T, n))

    w1 = np.random.rand(2, n1)
    w2 = np.random.rand(3, n2)
    w3 = np.random.rand(3, n3)

    fr[:,:,:n1] = factors[:,:,:2] @ w1 
    fr[:,:,n1:n1+n2] = factors[:,:,2:5] @ w2
    fr[:,:,n1+n2:] = factors[:,:,5:] @ w3

    spike_data = np.random.poisson(np.exp(fr))

    return spike_data, area_ind_list, fr, factors, stim, choice


#%%
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
from loader.data_loader import DatasetDataLoader

def make_rnn_loader(session_ind_list, batch_size):
    datasets = {'train': [], 'val': [], 'test': []}
    data_loader = {}
    num_neurons = []
    num_trials = {'train': [], 'val': [], 'test': []}
    area_ind_list_list = []
    record_info_list = []
    for session_ind in session_ind_list:

        n1 = np.random.randint(20, 30)
        n2 = np.random.randint(30, 40)
        n3 = np.random.randint(20, 30)

        K = np.random.randint(100, 200)

        spike_data, area_ind_list, fr, factors, stim, choice = rnn_data_loader(K, n1, n2, n3, T = 120)

        #randomly omit one region
        omit_region = np.random.randint(3)
        record_neuron_flag = (area_ind_list != omit_region)
        spike_data = spike_data[:,:,record_neuron_flag==1]
        area_ind_list = area_ind_list[record_neuron_flag==1]
        record_info = {'gt_n': [n1, n2, n3], 'omit_region': omit_region}

        record_info_list.append(record_info)

        #get train/valid/test trial_ind
        indices = np.arange(K)
        np.random.shuffle(indices)
        train_ind, val_ind, test_ind = np.split(indices, [int(.6*K), int(.8*K)])
        
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
    
    #breakpoint()
    return data_loader, num_neurons, datasets, area_ind_list_list, record_info_list