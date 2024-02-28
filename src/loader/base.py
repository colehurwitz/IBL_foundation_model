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

class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        max_pad = 5000,
        pad_to_ritght = True
    ) -> None:
        self.dataset = dataset
        self.max_pad = max_pad
        self.pad_to_ritght = pad_to_ritght

    def _preprocess(self, data):
        data = np.array(data['spikes_sparse_data'])
        if data.shape[0] > self.max_pad:
            data = data[:self.max_pad]
        else:
            if self.pad_to_ritght:
                data = _pad_seq_right_to_n(data, self.max_pad)
            else:
                data = _pad_seq_left_to_n(data, self.max_pad)
        return {"spikes_sparse_data": data}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self._preprocess(self.dataset[idx])