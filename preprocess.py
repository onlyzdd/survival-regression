import argparse
import os
import pickle

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def fit_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(np.vstack(X_train))
    with open('./models/standard_scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    return scaler


def transform(X, sample_ids, scaler, num_features, timesteps, output_dir):
    for i, sample_id in tqdm(enumerate(sample_ids)):
        x = X[i]
        df = pd.DataFrame(scaler.transform(x)) # apply scaler
        df = df.fillna(0) # impute with 0s
        df_zeros = pd.DataFrame(np.zeros((timesteps, num_features)), columns=range(num_features))
        df = pd.concat([df, df_zeros]) # padding and use first 90 rows
        df[:timesteps].to_csv(os.path.join(output_dir, sample_id), index=None, header=None)


def read_data(input_dir, sample_ids, num_features):
    X = []
    for sample_id in tqdm(sample_ids):
        x = pd.read_csv(os.path.join(input_dir, sample_id), sep='\t', header=None, names=range(num_features))
        x.index = [sample_id] * len(x)
        X.append(x)
    return X


def describe(X):
    return pd.concat(X).describe().T


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', type=str, default='tmp_data', help='Input directory')
    parser.add_argument('-o', '--output-dir', type=str, default='data', help='Output directory')
    parser.add_argument('--num-features', type=int, default=80, help='Number of features')
    parser.add_argument('--timesteps', type=int, default=90, help='Number of timesteps to use')
    args = parser.parse_args()
    input_dir, output_dir = args.input_dir, args.output_dir
    num_features, timesteps = args.num_features, args.timesteps

    train_ids = pd.read_csv('train_gs.dat', sep='\t', header=None)[0]
    train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)
    test_ids = pd.read_csv('test_gs.dat', sep='\t', header=None)[0]

    df_scores = pd.read_csv('train_target.txt', sep='\t', header=None, index_col=0)[1]
    
    X_train, X_val = [read_data(input_dir, sample_ids, num_features) for sample_ids in (train_ids, val_ids)]
    X_test = [] # skip test data for now

    scaler = fit_scaler(X_train)
    transform(X_train, train_ids, scaler, num_features, timesteps, output_dir)
    transform(X_val, val_ids, scaler, num_features, timesteps, output_dir)
    # transform(X_test, test_ids, scaler, num_features, timesteps, output_dir) # comment out

    # save score dicts
    df_scores.loc[train_ids].to_json(os.path.join(output_dir, 'train.json'))
    df_scores.loc[val_ids].to_json(os.path.join(output_dir, 'val.json'))
    # df_scores.loc[test_ids].to_json(os.path.join(output_dir, 'test.json')) # comment out
