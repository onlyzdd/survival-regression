import json
import os
from typing import TypeVar

import pandas as pd
import torch
from torch import dtype
from torch.utils.data import Dataset

T_co = TypeVar('T_co', covariant=True)


class DiskDataset(Dataset):
    def __init__(self, data_dir, score_file) -> None:
        self.data_dir = data_dir
        self.score_dict = json.load(open(score_file))
        self.sampleids = list(self.score_dict.keys())
        
    def __len__(self) -> int:
        return len(self.score_dict)

    def __getitem__(self, index: int) -> T_co:
        sampleid = self.sampleids[index]
        df = pd.read_csv(os.path.join(self.data_dir, sampleid), header=None)
        return torch.from_numpy(df.to_numpy()).float(), torch.Tensor([self.score_dict.get(sampleid)]).float() # data, label
