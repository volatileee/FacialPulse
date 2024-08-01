import numpy as np
import matplotlib.pyplot as plt
import torch
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
def predict_regression(model, data_iter, device):
    predictions = []
    model.to(device)
    model.eval()
    for X, _ in data_iter:
        X = X.to(device)
        output = model(X)
        prediction_batch = output.cpu().detach().numpy()
        predictions.append(prediction_batch)
    model.train()
    prediction_all = np.concatenate(predictions, axis=0)
    return prediction_all

def merge_video_prediction(mix_prediction, s2v, vc):

    prediction_video = []
    pre_count = {}
    for p, v_label in zip(mix_prediction, s2v):
        if v_label in pre_count:
            pre_count[v_label] += p
        else:
            pre_count[v_label] = p
    for key in pre_count.keys():
        prediction_video.append(pre_count[key] / vc[key])
    return prediction_video
