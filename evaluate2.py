import torch
from tqdm import tqdm

def calculate_mae(predict, target):
    return (torch.abs(predict - target)).mean().item()

def calculate_rmse(predict, target):
    return torch.sqrt(((predict - target) ** 2).mean()).item()

def evaluate(model, data_iter, device):
    model.eval()
    loss_sum, mae_sum, rmse_sum, samples_sum = 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for X, y in tqdm(data_iter, desc="Evaluation"):
            X = X.to(device)
            y = y.to(device)
            samples_num = X.shape[0]
            output = model(X)
            output = output.to(torch.float)
            loss = torch.nn.MSELoss()(output, y)
            loss_sum += loss.item() * samples_num
            mae_sum += calculate_mae(output, y) * samples_num
            rmse_sum += calculate_rmse(output, y) * samples_num
            samples_sum += samples_num

    avg_loss = loss_sum / samples_sum
    avg_mae = mae_sum / samples_sum
    avg_rmse = rmse_sum / samples_sum

    return avg_loss, avg_mae, avg_rmse
import argparse
import os
from os.path import join
import torch
from model import *
from Utils.dataset import Dataset
from Utils.logger import Logger
from Utils.metric import *
from configs.loader import load_yaml

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

    device = "cuda:1" if if_gpu and torch.cuda.is_available() else "cpu"

    dataset = Dataset(add_root=args_train["add_dataset_root"], name=dataset_name)

    logger = Logger()
    logger.register_status(dataset=dataset, device=device, branch_selection=branch_selection)
    logger.register_args(**args_train, **args_model)
    logger.print_logs_evaluation()

    if branch_selection == 'g1':
        model = Net(**args_model)
        model.load_state_dict(torch.load(join(add_weights, args_train["weights_name_g1"])))
    elif branch_selection == 'g2':
        model = Net(**args_model)
        model.load_state_dict(torch.load(join(add_weights, args_train["weights_name_g2"])))
    else:
        print("Unknown branch selection:", branch_selection, '. Please check and restart')
        return
    model.to(device)

    if branch_selection == 'g1':
        val_iter = dataset.load_data_val_g1(BLOCK_SIZE, BATCH_SIZE)
    elif branch_selection == 'g2':
        val_iter = dataset.load_data_val_g2(BLOCK_SIZE, BATCH_SIZE)
    else:
        print("Unknown branch selection:", branch_selection, '. Please check and restart')
        return

    avg_loss, avg_mae, avg_rmse = evaluate(model, val_iter, device)

    print(f"Avg Loss: {avg_loss:.4f}, Avg MAE: {avg_mae:.4f}, Avg RMSE: {avg_rmse:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluation codes.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-g', '--gpu', action='store_true',
                        help="If use the GPU(CUDA) for evaluation.")
    parser.add_argument('-d', '--dataset', type=str,
                        choices=['AVEC-org'],
                        default='AVEC-org',
                        help="Select the dataset used for evaluation. Valid selections: ['AVEC-org']")
    parser.add_argument('-b', '--branch', type=str,
                        choices=['g1', 'g2'],
                        default='g1',
                        help="Select which branch to be evaluated. Valid selections: ['g1', 'g2']")
    args = parser.parse_args()
    main(args)
