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

class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        pad_value = 0.,
        max_length = 5000,
        bin_size = 0.05,
        mask_ratio = 0.1,
        pad_to_right = True
    ) -> None:
        self.dataset = dataset
        self.pad_value = pad_value
        self.max_length = max_length
        self.bin_size = bin_size
        self.pad_to_right = pad_to_right
        self.mask_ratio = mask_ratio

    def _preprocess(self, data):
        spikes_sparse_data_list = [data['spikes_sparse_data']]
        spikes_sparse_indices_list = [data['spikes_sparse_indices']]
        spikes_sparse_indptr_list = [data['spikes_sparse_indptr']]
        spikes_sparse_shape_list = [data['spikes_sparse_shape']]

        # [bs, n_bin, n_spikes]
        binned_spikes_data = get_binned_spikes_from_sparse(spikes_sparse_data_list, 
                                                           spikes_sparse_indices_list, 
                                                           spikes_sparse_indptr_list, 
                                                           spikes_sparse_shape_list)
        # binned_spikes_data = np.einsum('bns->bsn', binned_spikes_data)
        # [n_spikes, n_neurons]]
        binned_spikes_data = binned_spikes_data[0]

        pad_length = 0

        seq_len = binned_spikes_data.shape[0]

        if seq_len > self.max_length:
            binned_spikes_data = binned_spikes_data[:self.max_length]
        else: 
            if self.pad_to_right:
                pad_length = self.max_length - seq_len
                binned_spikes_data = _pad_seq_right_to_n(binned_spikes_data, self.max_length, self.pad_value)
            else:
                pad_length = seq_len - self.max_length
                binned_spikes_data = _pad_seq_left_to_n(binned_spikes_data, self.max_length, self.pad_value)

        # add attention mask
        attention_mask = _attention_mask(self.max_length, pad_length).astype(np.int64)

        # add spikes timestamps [bs, n_spikes]
        # multiply by 100 to convert to int64
        spikes_timestamps = _spikes_timestamps(self.max_length, self.bin_size) * 100
        spikes_timestamps = spikes_timestamps.astype(np.int64)

        binned_spikes_data = binned_spikes_data.astype(np.float32)
        return {"binned_spikes_data": binned_spikes_data,
                "spikes_timestamps": spikes_timestamps,
                "attention_mask": attention_mask}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self._preprocess(self.dataset[idx])