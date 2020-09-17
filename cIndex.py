import argparse

import pandas as pd
from sksurv.metrics import concordance_index_censored


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=42)
    args = parser.parse_args()
    input_file = 'input_{}.txt'.format(args.seed)
    output_file = 'output_{}.txt'.format(args.seed)
    df = pd.read_csv(input_file, sep='\t', header=None, names=['time', 'trues', 'scores'])
    df['trues'] = df['trues'].astype(bool)
    cIndex = concordance_index_censored(df['trues'], df['time'], df['scores'])[0]
    print(cIndex)
