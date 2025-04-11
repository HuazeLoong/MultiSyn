import torch
import numpy as np
from math import sqrt
from torch import nn
from scipy import stats
from sklearn.metrics import auc, mean_absolute_error, roc_auc_score


def remove_nan_label(pred, truth):
    nan = torch.isnan(truth)
    truth = truth[~nan]
    pred = pred[~nan]

    return pred, truth


def roc_auc(pred, truth):
    return roc_auc_score(truth, pred)


def rmse(pred, truth):
    # print(f"pred type: {type(pred)}, truth type: {type(truth)}")
    pred_tensor = torch.tensor(pred, dtype=torch.float32)
    truth_tensor = torch.tensor(truth, dtype=torch.float32)

    return torch.sqrt(torch.mean(torch.square(pred_tensor - truth_tensor)))
    # return nn.functional.mse_loss(pred,truth)**0.5


def mae(pred, truth):
    return mean_absolute_error(truth, pred)


func_dict = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "mse": nn.MSELoss(),
    "rmse": rmse,
    "mae": mae,
    "crossentropy": nn.CrossEntropyLoss(),
    "bce": nn.BCEWithLogitsLoss(),
    "auc": roc_auc,
}


def get_func(fn_name):
    fn_name = fn_name.lower()
    return func_dict[fn_name]


def save_AUCs(AUCs, filename):
    with open(filename, "a") as f:
        f.write(",".join(map(str, AUCs)) + "\n")


def mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse
