import torch
import numpy as np
from math import sqrt
from torch import nn
from scipy import stats
from sklearn.metrics import auc, mean_absolute_error,roc_auc_score

def remove_nan_label(pred,truth):
    nan = torch.isnan(truth)
    truth = truth[~nan]
    pred = pred[~nan]

    return pred,truth

def roc_auc(pred,truth):
    return roc_auc_score(truth,pred)

def rmse(pred,truth):
    # print(f"pred type: {type(pred)}, truth type: {type(truth)}")
    pred_tensor = torch.tensor(pred, dtype=torch.float32)
    truth_tensor = torch.tensor(truth, dtype=torch.float32)
    
    return torch.sqrt(torch.mean(torch.square(pred_tensor - truth_tensor)))
    # return nn.functional.mse_loss(pred,truth)**0.5

def mae(pred,truth):
    return mean_absolute_error(truth,pred)

func_dict={'relu':nn.ReLU(),
           'sigmoid':nn.Sigmoid(),
           'mse':nn.MSELoss(),
           'rmse':rmse,
           'mae':mae,
           'crossentropy':nn.CrossEntropyLoss(),
           'bce':nn.BCEWithLogitsLoss(),
           'auc':roc_auc,
           }

def get_func(fn_name):
    fn_name = fn_name.lower()
    return func_dict[fn_name]

def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write(','.join(map(str, AUCs)) + '\n')
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci
