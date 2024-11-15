# %%
from utils.svoboda_data_utils import *
import pandas as pd
from pynwb import NWBHDF5IO
import seaborn as sns
import pickle
import torch
import numpy as np
import glob

#%%
# load session_info.csv as pandas dataframe
def load_session_data(session_ind):
    files = sorted(glob.glob('/mnt/smb/locker/miller-locker/users/jx2484/data/loaded_before/session_ind_*.pickle'))
    session_ind_list = [int(file.split('_')[-1].split('.')[0]) for file in files]
    if session_ind in session_ind_list:
        with open(files[session_ind_list.index(session_ind)], 'rb') as f:
            return pickle.load(f)

    path = '/mnt/smb/locker/miller-locker/users/jx2484/data/tables_and_infos/'

    session_info = pd.read_csv(path + 'session_info.csv')
    file_name = session_info['session_name'][session_ind]
    print(session_info.iloc[session_ind])

    io = NWBHDF5IO(path + file_name, mode="r")
    nwbfile = io.read()

    # get the spike train
    spike_data_combine_all, n_trial_all, licks_dict, behavior_dict, units, ids_sort_by_area, opto_type_sorted_dict, sorted_area, area_value, area_value_dict = main_get_spike_trains(nwbfile)

    ## balance trial number for left hit and right hit
    n_trial_balance = min(n_trial_all['hit'].values())
    n_trial_left = n_trial_all['hit']['left']

    spike_data_combine = spike_data_combine_all['hit']
    spike_data_balance = np.concatenate((spike_data_combine[:,:,:n_trial_balance], spike_data_combine[:,:,n_trial_left:n_trial_left+n_trial_balance]), axis=2)

    #load region_info_summary.pkl
    with open(path + 'region_info_summary.pkl', 'rb') as f:
        [brain_region_list, session_by_region, session_by_region_n, junk] = pickle.load(f)

    n_session_by_region = np.sum(session_by_region, axis=1)
    order_area = np.argsort(n_session_by_region)[::-1]
    areaoi_ind = order_area[:40] # this can be changed later; maybe select several areas we think is important
                                # now it's just the top 40 areas with most sessions

    # turn the area_value into a list of area index
    area_value = np.array(area_value)
    area_ind_list = np.zeros_like(area_value)
    for area, value in area_value_dict.items():
        area_ind = np.where(brain_region_list == area)[0][0]
        area_ind_list[area_value==value] = area_ind

    # get behavior data as D x T X K
    jaw_balance = np.concatenate((behavior_dict['hit']['left']['jaw'][:n_trial_balance,:,:], behavior_dict['hit']['right']['jaw'][:n_trial_balance,:,:]), axis=0)
    nose_balance = np.concatenate((behavior_dict['hit']['left']['nose'][:n_trial_balance,:,:], behavior_dict['hit']['right']['nose'][:n_trial_balance,:,:]), axis=0)
    tongue_balance = np.concatenate((behavior_dict['hit']['left']['tongue'][:n_trial_balance,:,:], behavior_dict['hit']['right']['tongue'][:n_trial_balance,:,:]), axis=0)

    behavior_balance = np.concatenate((jaw_balance, nose_balance, tongue_balance), axis=2)

    with open('/mnt/smb/locker/miller-locker/users/jx2484/data/loaded_before/session_ind_{}.pickle'.format(session_ind), 'wb') as f:
        pickle.dump([spike_data_balance, behavior_balance, area_ind_list, areaoi_ind, n_trial_balance], f)

    return spike_data_balance, behavior_balance, area_ind_list, areaoi_ind, n_trial_balance


#%%
#session_ind = 172
#spike_data, behavior_data, area_ind_list, areaoi_ind, n_trial_balance = load_session_data(session_ind)

# spike_data: N x T x 2*K
# behavior_data: 2*K x T x D 

#%%
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, spike_data, behavior_data, 
                 area_ind_list, areaoi_ind, 
                 session_ind):
        
        neuronoi_ind = np.array([], dtype=int)
        for area_ind in areaoi_ind:
            neuronoi_ind_tmp = np.where(area_ind_list == area_ind)[0]
            neuronoi_ind = np.concatenate((neuronoi_ind, neuronoi_ind_tmp))
            
        self.spike_data = np.swapaxes(spike_data[neuronoi_ind], 0, 2) # K x T x N
        self.area_ind_list = area_ind_list[neuronoi_ind]  # list of area_ind for each neuron
        self.areaoi_ind = areaoi_ind # area_ind of the area of interest
        self.session_ind = session_ind
        self.behavior_data = behavior_data # K x T x D

        K, T, N = self.spike_data.shape
        self.N = N
        self.T = T
        self.K = K

    def _preprocess_svoboda_data(self, idx):
        
        binned_spikes_data = self.spike_data[idx].astype(np.float32) # T x N
        time_attn_mask = np.ones((self.T,))
        space_attn_mask = np.ones((self.N,))

        spikes_timestamps = np.arange(self.T)  # maybe can be changed to behavior_dict[type]['ts'] later
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
                    "eid": self.session_ind
                }


    def __len__(self):
        return self.K
    
    def __getitem__(self, idx):
        return self._preprocess_svoboda_data(idx)

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
    
#%%
def make_loader(session_ind_list, batch_size):
    datasets = {'train': [], 'val': [], 'test': []}
    data_loader = {}
    num_neurons = []
    num_trials = {'train': [], 'val': [], 'test': []}
    area_ind_list_list = []
    for session_ind in session_ind_list:
        spike_data, behavior_data, area_ind_list, areaoi_ind, n_trial_balance = load_session_data(session_ind)

        #get train/valid/test trial_ind
        indices = np.arange(n_trial_balance)
        np.random.shuffle(indices)
        train_ind, val_ind, test_ind = np.split(indices, [int(.6*n_trial_balance), int(.8*n_trial_balance)])
        
        for ind, name in zip([train_ind, val_ind, test_ind], ['train', 'val', 'test']):

            dataset = BaseDataset(spike_data[:,:,ind], behavior_data[ind], area_ind_list, areaoi_ind, session_ind)
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
    return data_loader, num_neurons, datasets, areaoi_ind, area_ind_list_list
    
# data_loader, num_neurons, junk = make_loader([170,172], 12)

# breakpoint()

# for batch in data_loader['train']:
#     print(batch['spikes_data'].shape)
#     print(batch['target'].shape)
#     break

# breakpoint()

# %%
