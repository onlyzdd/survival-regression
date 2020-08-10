# README

Deep learning for survival analysis using PyTorch.

## Structure

- main.py: Code for main entry
- net.py: Code for LSTM model
- preprocess.py: Code for preprocessing
- dataset.py: Dataset for disk survival analysis

## Requirements

- python 3.6
- pytorch 1.3
- tqdm 4.43
- pandas 0.25
- numpy 1.16
- sci-kit learn 0.21

## Run

```
$ python main.py --help # show help info
$ python main.py # run code
```

## Todos

- Preprocessing:
  - Data imputation: lots of missing data especially at early times
  - Data normalization: Binarize categorical features and normalize other features
  - Determine the number of timesteps to be used
