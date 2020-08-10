import argparse
from glob import glob
import os

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess(X_train, X_val, output_dir):
    scaler = StandardScaler()
    scaler.fit(np.vstack(X_train).flatten())
    pass


def apply_scaler(X, scaler):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', type=str, default='tmp_data', help='Input directory')
    parser.add_argument('-o', '--output-dir', type=str, default='data', help='Output directory')
    args = parser.parse_args()
    input_dir, output_dir = args.input_dir, args.output_dir
    train_ids = pd.read_csv('train_gs.dat', sep='\t', header=None)[0]
    train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)
    X_train, X_val = [], []
    for sample_id in tqdm(train_ids):
        print(sample_id)
        x = pd.read_csv(os.path.join(input_dir, sample_id), sep='\t', header=None)
        X_train.append(x)
    print(len(X_train))