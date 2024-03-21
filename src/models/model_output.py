from dataclasses import dataclass
from typing import Optional

import torch


""" Base model output class. The models outputs should subclass this one. It is passed to the metric 
functions. Average loss will be computed dividing the sum of the losses by the sum of the number of 
examples over a series of calls to the model.
"""
@dataclass
class ModelOutput():
    loss: Optional[torch.FloatTensor] = None
    n_examples: Optional[torch.LongTensor] = None
    
    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__.keys()}