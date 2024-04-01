import torch
import numpy as np
from utils.dataset_utils import get_binned_spikes_from_sparse

def _pad_seq_right_to_n(
    seq: np.ndarray,
    n: int,
    pad_value: float = 0.
    ) -> np.ndarray:
    if n == seq.shape[0]:
        return seq
    return np.concatenate(
        [
            seq,
            np.ones(
                (
                    n-seq.shape[0],
                    *seq.shape[1:]
                )
            ) * pad_value,  
        ],
        axis=0,
    )

def _pad_seq_left_to_n(
    seq: np.ndarray,
    n: int,
    pad_value: float = 0.
    ) -> np.ndarray:
    if n == seq.shape[0]:
        return seq
    return np.concatenate(
        [
            np.ones(
                (
                    n-seq.shape[0],
                    *seq.shape[1:]
                )
            ) * pad_value,
            seq,
        ],
        axis=0,
    )

def _attention_mask(
    seq_length: int,
    pad_length: int,
    ) -> np.ndarray:
    mask = np.ones(seq_length)
    if pad_length:
        mask[-pad_length:] = 0
    else:
        mask[:pad_length] = 0
    return mask

def _spikes_timestamps(
    seq_length: int,
    bin_size: float = 0.02,
    ) -> np.ndarray:
    return np.arange(0, seq_length * bin_size, bin_size)

def _spikes_mask(
    seq_length: int,
    mask_ratio: float = 0.1,
    ) -> np.ndarray:
    # output 0/1
    return np.random.choice([0, 1], size=(seq_length,), p=[mask_ratio, 1-mask_ratio])

def _pad_spike_seq(
    seq: np.ndarray, 
    max_length: int,
    pad_to_right: bool = True,
    pad_value: float = 0.,
) -> np.ndarray:
    pad_length = 0
    seq_len = seq.shape[0]
    if seq_len > max_length:
        seq = seq[:max_length]
    else: 
        if pad_to_right:
            pad_length = max_length - seq_len
            seq = _pad_seq_right_to_n(seq, max_length, pad_value)
        else:
            pad_length = seq_len - max_length
            seq = _pad_seq_left_to_n(seq, max_length, pad_value)
    return seq, pad_length


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        target=None,
        pad_value = 0.,
        max_time_length = 5000,
        max_space_length = 100,
        n_neurons_per_patch = 64,
        bin_size = 0.05,
        mask_ratio = 0.1,
        pad_to_right = True,
        patching = False,
        sort_by_depth = False,
        dataset_name = "ibl",
    ) -> None:
        self.dataset = dataset
        self.target = target
        self.pad_value = pad_value
        self.patching = patching
        self.sort_by_depth = sort_by_depth
        self.max_time_length = max_time_length
        self.max_space_length = max_space_length
        self.n_neurons_per_patch = n_neurons_per_patch
        self.bin_size = bin_size
        self.pad_to_right = pad_to_right
        self.mask_ratio = mask_ratio
        self.dataset_name = dataset_name

    def _preprocess_h5_data(self, data, idx):
        spike_data, rates, _, _ = data
        spike_data, rates = spike_data[idx], rates[idx]
        # print(spike_data.shape, rates.shape)
        spike_data, pad_length = _pad_spike_seq(spike_data, self.max_time_length, self.pad_to_right, self.pad_value)
        # add attention mask
        attention_mask = _attention_mask(self.max_time_length, pad_length).astype(np.int64)
        # add spikes timestamps
        spikes_timestamps = _spikes_timestamps(self.max_time_length, 1)
        spikes_timestamps = spikes_timestamps.astype(np.int64)

        spike_data = spike_data.astype(np.float32)
        return {"spikes_data": spike_data, 
                "rates": rates, 
                "spikes_timestamps": spikes_timestamps, 
                "attention_mask": attention_mask}

    def _preprocess_ibl_data(self, data):
        spikes_sparse_data_list = [data['spikes_sparse_data']]
        spikes_sparse_indices_list = [data['spikes_sparse_indices']]
        spikes_sparse_indptr_list = [data['spikes_sparse_indptr']]
        spikes_sparse_shape_list = [data['spikes_sparse_shape']]

        # [bs, n_bin, n_spikes]
        binned_spikes_data = get_binned_spikes_from_sparse(spikes_sparse_data_list, 
                                                           spikes_sparse_indices_list, 
                                                           spikes_sparse_indptr_list, 
                                                           spikes_sparse_shape_list)

        if self.target is not None:
            target_behavior = np.array(data[self.target]).astype(np.float32)
            if self.target == 'choice':
                assert target_behavior != 0, "Invalid value for choice."
                target_behavior = np.array([0., 1.]) if target_behavior == 1 else np.array([1., 0.])
                target_behavior = target_behavior.astype(np.float32)
        else:
            target_behavior = np.array([np.nan])
            

        binned_spikes_data = binned_spikes_data[0]

        pad_time_length, pad_space_length = 0, 0

        num_time_steps, num_neurons = binned_spikes_data.shape
        max_num_neurons = self.max_space_length * self.n_neurons_per_patch

        if self.sort_by_depth:
            # sort neurons by depth on the probe
            neuron_depths = np.array(data['cluster_depths'])
            neuron_idxs = np.arange(num_neurons)
            sorted_neuron_idxs = [x for _, x in sorted(zip(neuron_depths, neuron_idxs))]
            binned_spikes_data = binned_spikes_data[:,sorted_neuron_idxs]
        
        # pad along time dimension
        if num_time_steps > self.max_time_length:
            binned_spikes_data = binned_spikes_data[:self.max_time_length]
        else: 
            if self.pad_to_right:
                pad_time_length = self.max_time_length - num_time_steps
                binned_spikes_data = _pad_seq_right_to_n(binned_spikes_data, self.max_time_length, self.pad_value)
            else:
                pad_time_length = num_time_steps - self.max_time_length
                binned_spikes_data = _pad_seq_left_to_n(binned_spikes_data, self.max_time_length, self.pad_value)
        if self.patching:
            # pad along space dimension
            if num_neurons > max_num_neurons:
                binned_spikes_data = binned_spikes_data[:,:max_num_neurons]
            else: 
                if self.pad_to_right:
                    pad_space_length = max_num_neurons - num_neurons
                    binned_spikes_data = _pad_seq_right_to_n(binned_spikes_data.T, max_num_neurons, self.pad_value)
                else:
                    pad_space_length = num_neurons - max_num_neurons
                    binned_spikes_data = _pad_seq_left_to_n(binned_spikes_data.T, max_num_neurons, self.pad_value)
                binned_spikes_data = binned_spikes_data.T
                    
            # group neurons into patches
            neuron_patches = np.ones(
                (self.max_time_length, self.max_space_length, self.n_neurons_per_patch)
            ) * self.pad_value    
            for patch_idx in range(self.max_space_length):
                neuron_patches[:, patch_idx, :] = \
                binned_spikes_data[:, patch_idx*self.n_neurons_per_patch:(patch_idx+1)*self.n_neurons_per_patch]
            # add space and time attention masks
            time_attention_mask = _attention_mask(self.max_time_length, pad_time_length).astype(np.int64)[:,None]
            time_attention_mask = np.repeat(time_attention_mask, self.max_space_length, 1)
            _space_attention_mask = _attention_mask(max_num_neurons, pad_space_length).astype(np.int64)[None,:]
            _space_attention_mask = np.repeat(_space_attention_mask, self.max_time_length, 0)

            # group space attention into patches
            space_attention_mask = np.ones((self.max_time_length, self.max_space_length)).astype(np.int64)        
            for patch_idx in range(self.max_space_length):
                if _space_attention_mask[:, patch_idx*self.n_neurons_per_patch:(patch_idx+1)*self.n_neurons_per_patch].sum() == 0:
                    space_attention_mask[:, patch_idx] = 0

            # add space and time steps
            spikes_timestamps = np.arange(self.max_time_length).astype(np.int64)[:,None]
            spikes_timestamps = np.repeat(spikes_timestamps, self.max_space_length, 1)
            spikes_spacestamps = np.arange(self.max_space_length).astype(np.int64)[None,:]
            spikes_spacestamps = np.repeat(spikes_spacestamps, self.max_time_length, 0)

            return {
                "neuron_patches": neuron_patches.astype(np.float32),
                "spikes_timestamps": spikes_timestamps,
                "spikes_spacestamps": spikes_spacestamps,
                "time_attention_mask": time_attention_mask,
                "space_attention_mask": space_attention_mask,
                "target": target_behavior
            }
        # add spikes timestamps [bs, n_spikes]
        # multiply by 100 to convert to int64
        # spikes_timestamps = _spikes_timestamps(self.max_time_length, self.bin_size) * 100
        # spikes_timestamps = spikes_timestamps.astype(np.int64)
        spikes_timestamps = np.arange(self.max_time_length).astype(np.int64)

        # add attention mask
        attention_mask = _attention_mask(self.max_time_length, pad_time_length).astype(np.int64)
        binned_spikes_data = binned_spikes_data.astype(np.float32)

        return {
            "spikes_data": binned_spikes_data,
            "spikes_timestamps": spikes_timestamps,
            "attention_mask": attention_mask,
            "target": target_behavior
        }
    
    def __len__(self):
        if "ibl" in self.dataset_name:
            return len(self.dataset)
        else:
            # get the length of the first tuple in the dataset
            return len(self.dataset[0])
    
    def __getitem__(self, idx):
        if "ibl" in self.dataset_name:
            return self._preprocess_ibl_data(self.dataset[idx])
        else:
            return self._preprocess_h5_data(self.dataset, idx)  

    