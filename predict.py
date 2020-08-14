import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from net import LSTM


if __name__ == "__main__":
    device = 'cpu'
    batch_size = 8
    model = LSTM(input_size=80, hidden_size=256, output_size=1).to(device)
    model.load_state_dict(torch.load('models/lstm.pth', map_location=device))
    model.eval()
    criterion = nn.MSELoss(reduction='sum')

    test_dict = json.load(open('data/val.json')) # use validation for now
    n = len(test_dict)
    outputs = []
    scores = []
    running_loss = 0
    for batch in tqdm(range(0, n, batch_size)):
        start = batch
        end = n if start + batch_size >= n else start + batch_size
        X, y = [], []
        for i in range(start, end):
            sample_id = list(test_dict.keys())[i]
            df = pd.read_csv(os.path.join('data', sample_id), header=None)
            X.append(df.to_numpy())
            y.append(test_dict[sample_id])
            scores.append(test_dict[sample_id])
        X = torch.from_numpy(np.array(X)).float()
        y = torch.from_numpy(np.array(y)[:, np.newaxis]).float()
        yy = model(X)
        running_loss += criterion(yy, y).item()
        outputs.append(yy.detach().numpy())
    outputs = np.vstack(outputs).flatten()
    print('MSELoss: %.4f' % (running_loss / n))
