import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm, trange

def calculate_mae(predict, target):
    return (torch.abs(predict - target)).mean().item()

def calculate_rmse(predict, target):
    return torch.sqrt(((predict - target) ** 2).mean()).item()

def evaluate(model, data_iter, device):
    mae_sum, samples_sum = 0.0, 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            samples_num = X.shape[0]
            output = model(X)
            mae_sum += calculate_mae(output, y) * samples_num
            samples_sum += samples_num
            rmse = calculate_rmse(output, y)
    model.train()
    mae = mae_sum / samples_sum
    return mae, rmse

def train_loop(model, train_iter, val_iter, optimizer, loss, epochs, device, add_weights_file):
    log_training_loss = []
    log_training_mae = []
    log_training_rmse = []
    log_val_mae = []
    log_val_rmse = []
    best_val_mae = float('inf')

    model.to(device)
    model.train()
    for epoch in trange(1, epochs + 1):
        loss_sum, mae_sum, rmse_sum, samples_sum = 0.0, 0.0, 0.0, 0
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            samples_num = X.shape[0]
            output = model(X)
            output = output.to(torch.float)
            l = loss(output.float(), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            loss_sum += l.item() * samples_num
            mae_sum += calculate_mae(output, y) * samples_num
            rmse_sum += calculate_rmse(output, y) * samples_num
            samples_sum += samples_num

        train_mae = mae_sum / samples_sum
        train_rmse = rmse_sum / samples_sum
        val_mae, val_rmse = evaluate(model, val_iter, device)

        if val_mae <= best_val_mae:
            save_hint = "save the model to {}".format(add_weights_file)
            torch.save(model.state_dict(), add_weights_file)
            best_val_mae = val_mae
        else:
            save_hint = ""

        tqdm.write("epoch:{}, loss:{:.4}, train_mae:{:.4}, train_rmse:{:.4}, "
                   "val_mae:{:.4}, val_rmse:{:.4}, best_val_mae:{:.4}  "
                   .format(epoch, loss_sum / samples_sum, train_mae, train_rmse, val_mae, val_rmse, best_val_mae)
                   + save_hint)

        log_training_loss.append(loss_sum / samples_sum)
        log_training_mae.append(train_mae)
        log_training_rmse.append(train_rmse)
        log_val_mae.append(val_mae)
        log_val_rmse.append(val_rmse)

    log = {"loss": log_training_loss, "mae_train": log_training_mae, "rmse_train": log_training_rmse,
           "mae_val": log_val_mae, "rmse_val": log_val_rmse}
    return log
import argparse
import os
from tqdm import tqdm, trange
from torch import optim
from os.path import join
from model import Net
from Utils.logger import Logger
from Utils.metric import calculate_mae, calculate_rmse
from Utils.dataset import Dataset
from configs.loader import load_yaml
import torch

def main(args):
    if_gpu = args.gpu
    dataset_name = args.dataset
    branch_selection = args.branch

    args_model = load_yaml("configs/args_model.yaml")
    args_train = load_yaml("configs/args_train.yaml")

    BLOCK_SIZE = args_train["BLOCK_SIZE"]
    BATCH_SIZE = args_train["BATCH_SIZE"]

    add_weights = args_train["add_weights"]
    if not os.path.exists(add_weights):
        os.makedirs(add_weights)

    EPOCHS_g1 = args_train["EPOCHS_g1"]
    LEARNING_RATE_g1 = args_train["LEARNING_RATE_g1"]
    weights_name_g1 = args_train["weights_name_g1"]

    EPOCHS_g2 = args_train["EPOCHS_g2"]
    LEARNING_RATE_g2 = args_train["LEARNING_RATE_g2"]
    weights_name_g2 = args_train["weights_name_g2"]

    if if_gpu:
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
    else:
        device = 'cpu'

    dataset = Dataset(add_root=args_train["add_dataset_root"], name=dataset_name)

    logger = Logger()
    logger.register_status(dataset=dataset, device=device, branch_selection=branch_selection)
    logger.register_args(**args_train, **args_model)
    logger.print_logs_training()

    train_iter_A = None
    train_iter_B = None
    val_iter_A = None
    val_iter_B = None
    if branch_selection == 'g1':
        train_iter_A = dataset.load_data_train_g1(BLOCK_SIZE, BATCH_SIZE)
        val_iter_A = dataset.load_data_val_g1(BLOCK_SIZE, BATCH_SIZE)
    elif branch_selection == 'g2':
        train_iter_B = dataset.load_data_train_g2(BLOCK_SIZE, BATCH_SIZE)
        val_iter_B = dataset.load_data_val_g2(BLOCK_SIZE, BATCH_SIZE)
    elif branch_selection == 'all':
        train_iter_A, train_iter_B = dataset.load_data_train_all(BLOCK_SIZE, BATCH_SIZE)
        val_iter_A, val_iter_B = dataset.load_data_val_all(BLOCK_SIZE, BATCH_SIZE)
    else:
        print("Unknown branch selection:", branch_selection, '. Please check and restart')
        return

    if branch_selection == 'g1' or branch_selection == 'all':
        assert train_iter_A is not None and val_iter_A is not None
        g1 = Net(**args_model)
        optimizer = optim.Adam(g1.parameters(), lr=LEARNING_RATE_g1)
        loss = torch.nn.SmoothL1Loss()
        add_weights_file = join(add_weights, weights_name_g1)
        log_g1 = train_loop(g1, train_iter_A, val_iter_A, optimizer, loss, EPOCHS_g1, device, add_weights_file)

    if branch_selection == 'g2' or branch_selection == 'all':
        assert train_iter_B is not None and val_iter_B is not None
        g2 = Net(**args_model)
        optimizer = optim.Adam(g2.parameters(), lr=LEARNING_RATE_g2)
        loss = torch.nn.SmoothL1Loss()
        add_weights_file = join(add_weights, weights_name_g2)
        log_g2 = train_loop(g2, train_iter_B, val_iter_B, optimizer, loss, EPOCHS_g2, device, add_weights_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training codes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-g', '--gpu', action='store_true', help="If use the GPU(CUDA) for training.")
    parser.add_argument('-d', '--dataset', type=str, choices=['AVEC-org'], default='AVEC-org',
                        help="Select the dataset used for training. Valid selections: ['AVEC-org']")
    parser.add_argument('-b', '--branch', type=str, choices=['g1', 'g2', 'all'], default='all',
                        help="Select which branch to be trained. Valid selections: ['g1', 'g2', 'all']")
    args = parser.parse_args()
    main(args)
