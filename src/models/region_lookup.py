import numpy as np
import torch
from torch import nn
from iblatlas.regions import BrainRegions
from utils.config_utils import DictConfig

"""
Work in progress. Create a global lookup table for all brain regions.
"""

class RegionLookup(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.brain_regions = BrainRegions().acronym
        self.region_to_indx = {r: i for i,r in enumerate(self.brain_regions, start=0)}
        self.indx_to_region = {v: k for k,v in self.region_to_indx.items()}
        self.max_region_indx = len(self.brain_regions) 

    def forward(
        self, 
        neuron_regions: np.ndarray
    ):
        region_indx = torch.stack(
            [
            torch.tensor(
                [self.region_to_indx[r] for r in row if r != 'nan'], 
                dtype=torch.int64
            ) for row in neuron_regions
            ], dim=0
        )
        return region_indx

    def lookup_regions(
        self,
        region_indx: torch.LongTensor 
    ):
        regions = torch.stack(
            [
            torch.tensor(
                [self.indx_to_region[r] for r in row if r != 'nan'], 
                dtype=torch.int64
            ) for row in region_indx
            ], dim=0
        )
        return regions
            
