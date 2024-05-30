import torch
import pickle
import numpy as np
from utils.dataset_utils import get_binned_spikes_from_sparse
from torch.utils.data.sampler import Sampler
from typing import List, Optional, Tuple, Dict
from torch.utils.data import Dataset

def _pad_seq_right_to_n(
    seq: np.ndarray,
    n: int,
    pad_value: float = 0.
    ) -> np.ndarray:
    if n == len(seq):
        return seq
    return np.concatenate(
        [
            seq,
            np.ones(
                (
                    n-len(seq),
                    *seq[0].shape
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
    if n == len(seq):
        return seq
    return np.concatenate(
        [
            np.ones(
                (
                    n-len(seq),
                    *seq[0].shape
                )
            ) * pad_value,
            seq,
        ],
        axis=0,
    )

def _wrap_pad_temporal_right_to_n(
    seq: np.ndarray,
    n: int
    ) -> np.ndarray:
    # input shape is [n_time_steps, n_neurons]
    # pad along time dimension, wrap around along space dimension
    if n == len(seq):
        return seq
    return np.pad(
        seq,
        ((0, n-seq.shape[0]), (0, 0)),
        mode='wrap'
    )
    
def _wrap_pad_neuron_up_to_n(
    seq: np.ndarray,
    n: int
    ) -> np.ndarray:
    # input shape is [n_time_steps, n_neurons]
    # pad along neuron dimension, wrap around along time dimension
    if n == len(seq[0]):
        return seq
    return np.pad(
        seq,
        ((0, 0), (0, n-seq.shape[1])),
        mode='wrap'
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



def get_length_grouped_indices(lengths, batch_size, shuffle=True, mega_batch_mult=None, generator=None):
    # Default for mega_batch_mult: 50 or the number to get 4 megabatches, whichever is smaller.
    if mega_batch_mult is None:
        mega_batch_mult = min(len(lengths) // (batch_size * 4), 50)
        # Just in case, for tiny datasets
        if mega_batch_mult == 0:
            mega_batch_mult = 1

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    if shuffle:
        indices = torch.randperm(len(lengths), generator=generator)
    else:
        indices = torch.arange(len(lengths))
    megabatch_size = mega_batch_mult * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [list(sorted(megabatch, key=lambda i: lengths[i], reverse=True)) for megabatch in megabatches]

    # The rest is to get the biggest batch first.
    # Since each megabatch is sorted by descending length, the longest element is the first
    megabatch_maximums = [lengths[megabatch[0]] for megabatch in megabatches]
    max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
    # Switch to put the longest element in first position
    megabatches[0][0], megabatches[max_idx][0] = megabatches[max_idx][0], megabatches[0][0]

    return sum(megabatches, [])



def get_length_grouped_indices_stitched(lengths, batch_size, generator=None):
    # sort indices by length
    sorted_indices = np.argsort(lengths)
    # random indices in same length group
    group_indicies = []
    group_lengths = []
    group = []
    for i, idx in enumerate(sorted_indices):
        if i == 0:
            group.append(idx)
            group_lengths.append(lengths[idx])
        elif lengths[idx] == group_lengths[-1]:
            group.append(idx)
        else:
            group_indicies.append(group)
            group = [idx]
            group_lengths.append(lengths[idx])
    group_indicies.append(group)
    group_indicies = sum(group_indicies,[])
    # makke group_indice a multiple of batch_size
    batch_group_indicies = []
    for i in range(0, len(group_indicies), batch_size):
        batch_group_indicies.append(group_indicies[i:i+batch_size])
    if generator is not None:
        generator.shuffle(batch_group_indicies)
    else:
        np.random.shuffle(batch_group_indicies)
    batch_group_indicies = sum(batch_group_indicies, [])
    batch_group_indicies = [int(i) for i in batch_group_indicies]
    return batch_group_indicies


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        lengths: Optional[List[int]] = None,
        shuffle: Optional[bool] = True,
        model_input_name: Optional[str] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.model_input_name = model_input_name if model_input_name is not None else "input_ids"
        if lengths is None:
            if not isinstance(dataset[0], dict) or self.model_input_name not in dataset[0]:
                raise ValueError(
                    "Can only automatically infer lengths for datasets whose items are dictionaries with an "
                    f"'{self.model_input_name}' key."
                )
            lengths = [len(feature[self.model_input_name]) for feature in dataset]
        self.lengths = lengths
        self.shuffle = shuffle

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, self.shuffle)
        return iter(indices)



class LengthStitchGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        lengths: Optional[List[int]] = None,
        model_input_name: Optional[str] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.model_input_name = model_input_name if model_input_name is not None else "input_ids"
        if lengths is None:
            if not isinstance(dataset[0], dict) or model_input_name not in dataset[0]:
                raise ValueError(
                    "Can only automatically infer lengths for datasets whose items are dictionaries with an "
                    f"'{self.model_input_name}' key."
                )
            lengths = [len(feature[self.model_input_name]) for feature in dataset]
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices_stitched(self.lengths, self.batch_size)
        return iter(indices)



class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        target=None,
        pad_value = 0.,
        max_time_length = 5000,
        max_space_length = 1000,
        bin_size = 0.05,
        mask_ratio = 0.1,
        pad_to_right = True,
        sort_by_depth = False,
        sort_by_region = False,
        load_meta = False,
        brain_region = 'all',
        dataset_name = "ibl",
        stitching = False,
        use_nemo = False,
    ) -> None:
        self.dataset = dataset
        self.target = target
        self.pad_value = pad_value
        self.sort_by_depth = sort_by_depth
        self.sort_by_region = sort_by_region
        self.max_time_length = max_time_length
        self.max_space_length = max_space_length
        self.bin_size = bin_size
        self.pad_to_right = pad_to_right
        self.mask_ratio = mask_ratio
        self.brain_region = brain_region
        self.load_meta = load_meta
        self.dataset_name = dataset_name
        self.stitching = stitching
        self.use_nemo = use_nemo

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

        if self.use_nemo:
            neuron_uuids = np.array(data['cluster_uuids']).astype('str')
            with open('data/MtM_unit_embed.pkl','rb') as file:
                nemo_data = pickle.load(file)
            nemo_uuids = nemo_data['uuids']
            nemo_rep = np.concatenate((nemo_data['wvf_rep'], nemo_data['acg_rep']), axis=1)
            include_uuids = np.intersect1d(neuron_uuids, nemo_uuids)
            nemo_rep = nemo_rep[np.argwhere(np.array([1 if uuid in include_uuids else 0 for uuid in nemo_uuids]).flatten() == 1).astype(np.int64)].squeeze()
            include_neuron_ids = np.argwhere(np.array([1 if uuid in include_uuids else 0 for uuid in neuron_uuids]).flatten() == 1).astype(np.int64)
            self.max_space_length = len(include_neuron_ids)
        else:
            include_neuron_ids = np.ones(binned_spikes_data.shape[-1]).flatten().astype(np.int64)
            nemo_rep = np.array([np.nan])
        
        binned_spikes_data = binned_spikes_data[:,include_neuron_ids].squeeze()

        if self.load_meta:
            if 'cluster_depths' in data:
                neuron_depths = np.array(data['cluster_depths']).astype(np.float32)
                neuron_depths = neuron_depths[include_neuron_ids].squeeze()
            else:
                neuron_depths = np.array([np.nan])
            neuron_regions = np.array(data['cluster_regions']).astype('str')
            neuron_regions = neuron_regions[include_neuron_ids].squeeze()
        else:
            neuron_depths = neuron_regions = np.array([np.nan])
            
        if self.load_meta & (self.brain_region != 'all'):
            region_idxs = np.argwhere(neuron_regions == self.brain_region)
            binned_spikes_data = binned_spikes_data[:,region_idxs].squeeze()
            neuron_regions = neuron_regions[region_idxs]
            if self.sort_by_depth:
                neuron_depths = neuron_depths[region_idxs]   
            if self.use_nemo:
                nemo_rep = nemo_rep[region_idxs]

        pad_time_length, pad_space_length = 0, 0

        num_time_steps, num_neurons = binned_spikes_data.shape

        if self.load_meta:
            neuron_idxs = np.arange(num_neurons)
            assert (self.sort_by_depth and self.sort_by_region) == False, "Can only sort either by depth or neuron."
            if self.sort_by_depth:
                sorted_neuron_idxs = [x for _, x in sorted(zip(neuron_depths, neuron_idxs))]
            elif self.sort_by_region:
                sorted_neuron_idxs = [x for _, x in sorted(zip(neuron_regions, neuron_idxs))]
            else:
                sorted_neuron_idxs = neuron_idxs.copy()
            binned_spikes_data = binned_spikes_data[:,sorted_neuron_idxs]
            neuron_depths = neuron_depths[sorted_neuron_idxs]
            neuron_regions = neuron_regions[sorted_neuron_idxs]
            if self.use_nemo:
                nemo_rep = nemo_rep[sorted_neuron_idxs]

        neuron_regions = list(neuron_regions)
        
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

        if not self.stitching:
            # pad along space dimension
            if num_neurons > self.max_space_length:
                binned_spikes_data = binned_spikes_data[:,:self.max_space_length]
                neuron_depths = neuron_depths[:self.max_space_length]
                neuron_regions = neuron_regions[:self.max_space_length]
                if self.use_nemo:
                    nemo_rep = nemo_rep[:self.max_space_length]
            else: 
                if self.pad_to_right:
                    pad_space_length = self.max_space_length - num_neurons
                    binned_spikes_data = _pad_seq_right_to_n(binned_spikes_data.T, self.max_space_length, self.pad_value)
                    # binned_spikes_data = _wrap_pad_neuron_up_to_n(binned_spikes_data, self.max_space_length).T
                    neuron_depths = _pad_seq_right_to_n(neuron_depths, self.max_space_length, np.nan)
                    neuron_regions = _pad_seq_right_to_n(neuron_regions, self.max_space_length, np.nan)
                    if self.use_nemo:
                        nemo_rep = _pad_seq_right_to_n(nemo_rep, self.max_space_length, np.nan)
                else:
                    pad_space_length = num_neurons - self.max_space_length
                    binned_spikes_data = _pad_seq_left_to_n(binned_spikes_data.T, self.max_space_length, self.pad_value)
                    neuron_depths = _pad_seq_left_to_n(neuron_depths, self.max_space_length, np.nan)
                    neuron_regions = _pad_seq_left_to_n(neuron_regions, self.max_space_length, np.nan)
                    if self.use_nemo:
                        nemo_rep = _pad_seq_left_to_n(nemo_rep, self.max_space_length, np.nan)
                binned_spikes_data = binned_spikes_data.T
            spikes_spacestamps = np.arange(self.max_space_length).astype(np.int64)
            space_attn_mask = _attention_mask(self.max_space_length, pad_space_length).astype(np.int64)
        else:
            spikes_spacestamps = np.arange(num_neurons).astype(np.int64)
            space_attn_mask = _attention_mask(num_neurons, 0).astype(np.int64)
                
        spikes_timestamps = np.arange(self.max_time_length).astype(np.int64)
        
        # add attention mask
        time_attn_mask = _attention_mask(self.max_time_length, pad_time_length).astype(np.int64)
        binned_spikes_data = binned_spikes_data.astype(np.float32)
        
        return {
            "spikes_data": binned_spikes_data,
            "time_attn_mask": time_attn_mask,
            "space_attn_mask": space_attn_mask,
            "spikes_timestamps": spikes_timestamps,
            "spikes_spacestamps": spikes_spacestamps,
            "target": target_behavior,
            "neuron_depths": neuron_depths, 
            "neuron_regions": list(neuron_regions),
            "eid": data['eid'],
            "nemo_rep": nemo_rep,
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
 

    
