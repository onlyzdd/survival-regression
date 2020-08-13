import argparse
import os
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DiskDataset
from net import LSTM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='tmp_data', help='Directory for data dir')
    parser.add_argument('--score-file', type=str, default='train_target.txt', help='Path to score file')

    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--model', type=str, default='lstm', help='Model: lstm')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Num of workers to load data')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--use-gpu', default=False, action='store_true', help='Use GPU')
    parser.add_argument('--model-path', type=str, default='models', help='Path to saved model')

    return parser.parse_args()


def train(dataloader, n, model, epoch, args, criterion, optimizer, device):
    print('Training epoch', epoch, '...')
    model.train()
    running_loss = 0
    for _, (data, score) in enumerate(tqdm(dataloader)):
        data, score = data.to(device), score.to(device)
        output = model(data)
        output = torch.clamp(output, 0, 1)
        loss = criterion(output, score)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('MSELoss: %.4f' % (running_loss / n))


def evaluate(dataloader, n, model, args, criterion, device):
    print('Validating...')
    model.eval()
    running_loss = 0
    for _, (data, score) in enumerate(tqdm(dataloader)):
        data, score = data.to(device), score.to(device)
        output = model(data) # torch.clamp to ensure between 0 and 1
        output = torch.clamp(output, 0, 1)
        loss = criterion(output, score)
        running_loss += loss.item()
    print('MSELoss: %.4f' % (running_loss / n))
    if args.phase == 'train' and running_loss < args.best_metric:
        args.best_metric = running_loss
        torch.save(model.state_dict(), args.model_path)


if __name__ == "__main__":
    args = parse_args()
    args.best_metric = float('inf')
    data_dir =  args.data_dir
    train_score_file = args.score_file
    val_score_file = args.score_file
    test_score_file = args.score_file
    args.model_path = os.path.join(args.model_path, '%s.pth' % args.model)
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = 'cpu'
    model = LSTM(input_size=6, hidden_size=256, output_size=1).to(device)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    if args.phase == 'train':
        train_dataset = DiskDataset(data_dir, train_score_file)
        val_dataset = DiskDataset(data_dir, val_score_file)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
        for epoch in range(args.epochs):
            train(train_loader, len(train_dataset), model, epoch, args, criterion, optimizer, device)
            evaluate(val_loader, len(val_dataset), model, args, criterion, device)
    else:
        model.load_state_dict(torch.load(args.model_path))
        test_dataset = DiskDataset(data_dir, test_score_file)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
        evaluate(test_loader, len(test_dataset), model, args, criterion, device)
