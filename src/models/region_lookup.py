from iblatlas.regions import BrainRegions

"""
Work in progress. Create a global lookup table for all brain regions.
"""

class RegionLookup:
    def __init__(self):
        self.brain_regions = BrainRegions().acronym
        self.region_to_indx = {r: i for i,r in enumerate(self.brain_regions)}
        self.indx_to_region = {v: k for k,v in self.region_to_indx.items()}

    def lookup(self, regions):
        pass
            
