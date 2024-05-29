import torch
import numpy as np
from src.utils.dataset_utils import get_binned_spikes_from_sparse
from torch.utils.data.sampler import Sampler


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
                    n - len(seq),
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
                    n - len(seq),
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
        ((0, n - seq.shape[0]), (0, 0)),
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
        ((0, 0), (0, n - seq.shape[1])),
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
    return np.random.choice([0, 1], size=(seq_length,), p=[mask_ratio, 1 - mask_ratio])


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
            target=None,  # target behavior
            pad_value=0.,
            max_time_length=5000,
            max_space_length=1000,
            bin_size=0.05,
            mask_ratio=0.1,
            pad_to_right=True,
            sort_by_depth=False,
            sort_by_region=False,
            load_meta=False,
            brain_region='all',
            dataset_name="ibl",
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

        if self.load_meta:
            if 'cluster_depths' in data:
                neuron_depths = np.array(data['cluster_depths']).astype(np.float32)
            else:
                neuron_depths = np.array([np.nan])
            neuron_regions = np.array(data['cluster_regions']).astype('str')
        else:
            neuron_depths = neuron_regions = np.array([np.nan])

        if self.load_meta & (self.brain_region != 'all'):
            # only load neurons from a given brain region
            # this is for NDT2 since not enough RAM to load all neurons  
            region_idxs = np.argwhere(neuron_regions == self.brain_region)
            binned_spikes_data = binned_spikes_data[:, region_idxs].squeeze()
            neuron_regions = neuron_regions[region_idxs]
            if self.sort_by_depth:
                neuron_depths = neuron_depths[region_idxs]

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
            binned_spikes_data = binned_spikes_data[:, sorted_neuron_idxs]
            neuron_depths = neuron_depths[sorted_neuron_idxs]
            neuron_regions = neuron_regions[sorted_neuron_idxs]

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

        # pad along space dimension
        if num_neurons > self.max_space_length:
            binned_spikes_data = binned_spikes_data[:, :self.max_space_length]
            neuron_depths = neuron_depths[:self.max_space_length]
            neuron_regions = neuron_regions[:self.max_space_length]
        else:
            if self.pad_to_right:
                pad_space_length = self.max_space_length - num_neurons
                binned_spikes_data = _pad_seq_right_to_n(binned_spikes_data.T, self.max_space_length, self.pad_value)
                # binned_spikes_data = _wrap_pad_neuron_up_to_n(binned_spikes_data, self.max_space_length).T
                neuron_depths = _pad_seq_right_to_n(neuron_depths, self.max_space_length, np.nan)
                neuron_regions = _pad_seq_right_to_n(neuron_regions, self.max_space_length, np.nan)
            else:
                pad_space_length = num_neurons - self.max_space_length
                binned_spikes_data = _pad_seq_left_to_n(binned_spikes_data.T, self.max_space_length, self.pad_value)
                neuron_depths = _pad_seq_left_to_n(neuron_depths, self.max_space_length, np.nan)
                neuron_regions = _pad_seq_left_to_n(neuron_regions, self.max_space_length, np.nan)
            binned_spikes_data = binned_spikes_data.T

        spikes_timestamps = np.arange(self.max_time_length).astype(np.int64)
        spikes_spacestamps = np.arange(self.max_space_length).astype(np.int64)

        # add attention mask
        time_attn_mask = _attention_mask(self.max_time_length, pad_time_length).astype(np.int64)
        space_attn_mask = _attention_mask(self.max_space_length, pad_space_length).astype(np.int64)
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
