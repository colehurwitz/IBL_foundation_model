import torch
import numpy as np

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
    seq: np.ndarray,
    pad_length: int,
    ) -> np.ndarray:
    mask = np.ones_like(seq)
    if pad_length:
        mask[-pad_length:] = 0
    else:
        mask[:pad_length] = 0
    return mask

class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        pad_value = 0.,
        max_length = 5000,
        pad_to_right = True
    ) -> None:
        self.dataset = dataset
        self.pad_value = pad_value
        self.max_length = max_length
        self.pad_to_right = pad_to_right

    def _preprocess(self, data):
        data = np.array(data['spikes_sparse_data'])
        
        pad_length = 0

        if data.shape[0] > self.max_length:
            data = data[:self.max_length]
        else: 
            if self.pad_to_right:
                pad_length = self.max_length - data.shape[0]
                data = _pad_seq_right_to_n(data, self.max_length, self.pad_value)
            else:
                pad_length = data.shape[0] - self.max_length
                data = _pad_seq_left_to_n(data, self.max_length, self.pad_value)

        # add attention mask
        attention_mask = _attention_mask(data, pad_length)

        return {"spikes_sparse_data": data,
                "attention_mask": attention_mask}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self._preprocess(self.dataset[idx])