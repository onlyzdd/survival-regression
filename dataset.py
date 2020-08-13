import os
from typing import TypeVar

import pandas as pd
import torch
from torch.utils.data import Dataset

T_co = TypeVar('T_co', covariant=True)


class DiskDataset(Dataset):
    def __init__(self, data_dir, score_file) -> None:
        self.data_dir = data_dir
        self.score_dict = pd.read_csv(score_file, sep='\t', header=None, index_col=0)[1].to_dict()
        self.sampleids = list(self.score_dict.keys())
        
    def __len__(self) -> int:
        return len(self.score_dict)

    def __getitem__(self, index: int) -> T_co:
        sampleid = self.sampleids[index]
        # TODO Please make sure the inputs are preprocessed, i.e. missing data imputation, data scaling, padding and cropping
        df = pd.read_csv(os.path.join(self.data_dir, sampleid), header=None)[1:]
        return torch.rand((5, 6)), torch.rand(1) # data, label
