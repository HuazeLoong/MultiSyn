import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from const import *
from random import shuffle
from model import *
from sklearn import metrics
from dataset import MyTestDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
    balanced_accuracy_score,
)
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    roc_auc_score,
    precision_score,
)

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch):
    """
    During the model training process, the graph data and cell features of the two drugs are obtained batch by batch,
    and forward propagation + back propagation + optimization is performed.
    """
    print("Training on {} samples...".format(len(drug1_loader_train.dataset)))
    model.train()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    total_loss = 0.0

    for batch_idx, data in enumerate(zip(drug1_loader_train, drug2_loader_train)):
        data1 = data[0]
        data2 = data[1]
        data1 = data1.to(device)
        data2 = data2.to(device)
        y = data1.y.view(-1, 1).long().to(device)
        y = y.squeeze(1)

        optimizer.zero_grad()
        output = model(data1, data2)
        loss = loss_fn(output, y)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

        # softmax outputs predicted scores and predicted labels
        ys = F.softmax(output, 1).to("cpu").data.numpy()
        predicted_labels = list(map(lambda x: np.argmax(x), ys))
        predicted_scores = list(map(lambda x: x[1], ys))
        total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
        total_prelabels = torch.cat(
            (total_prelabels, torch.Tensor(predicted_labels)), 0
        )
        total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)

        if batch_idx % LOG_INTERVAL == 0:
            print(
                "Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data1.y),
                    len(drug1_loader_train.dataset),
                    100.0 * batch_idx / len(drug1_loader_train),
                    loss.item(),
                )
            )


def predicting(model, device, drug1_loader_test, drug2_loader_test):
    """
    Model prediction process, used in the evaluation phase.
    Returns true label, prediction score, and predicted label.
    """
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()

    print("Make prediction for {} samples...".format(len(drug1_loader_test.dataset)))
    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1 = data[0]
            data2 = data[1]
            data1 = data1.to(device)
            data2 = data2.to(device)
            output = model(data1, data2)

            ys = F.softmax(output, 1).to("cpu").data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat(
                (total_prelabels, torch.Tensor(predicted_labels)), 0
            )
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)

    return (
        total_labels.numpy().flatten(),
        total_preds.numpy().flatten(),
        total_prelabels.numpy().flatten(),
    )


def compute_preformence(T, S, Y, best_auc, file):
    """Calculate multiple classification metrics and save the result corresponding to the best AUC."""
    AUC = roc_auc_score(T, S)
    precision, recall, threshold = metrics.precision_recall_curve(T, S)
    PR_AUC = metrics.auc(recall, precision)
    BACC = balanced_accuracy_score(T, Y)
    tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
    TPR = tp / (tp + fn)
    PREC = precision_score(T, Y)
    ACC = accuracy_score(T, Y)
    KAPPA = cohen_kappa_score(T, Y)
    recall = recall_score(T, Y)
    F1 = f1_score(T, Y)

    AUCs = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, recall, F1]
    if best_auc < AUC:
        save_AUCs(AUCs, file)
        best_auc = AUC
    return best_auc, AUC


modeling = MultiSyn

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
LR = 0.0001
LOG_INTERVAL = 20
NUM_EPOCHS = 400

print("Learning rate: ", LR)
print("Epochs: ", NUM_EPOCHS)
datafile = "drugcom_12415"

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("The code uses GPU...")
else:
    device = torch.device("CPU")
    print("The code uses CPU!!!")

drug1_data = MyTestDataset(root=DATAS_DIR, dataset=datafile + "_drug1")
drug2_data = MyTestDataset(root=DATAS_DIR, dataset=datafile + "_drug2")
lenth = len(drug1_data)
pot = int(lenth / 5)
print("lenth", lenth)
print("pot", pot)

# 5-fold random split
random_num = random.sample(range(0, lenth), lenth)
for i in range(5):
    # Construct DataLoader for training and test sets
    test_num = random_num[pot * i : pot * (i + 1)]
    train_num = random_num[: pot * i] + random_num[pot * (i + 1) :]

    drug1_data_train = drug1_data[train_num]
    drug1_data_test = drug1_data[test_num]
    drug1_loader_train = DataLoader(
        drug1_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None
    )
    drug1_loader_test = DataLoader(
        drug1_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None
    )

    drug2_data_test = drug2_data[test_num]
    drug2_data_train = drug2_data[train_num]
    drug2_loader_train = DataLoader(
        drug2_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None
    )
    drug2_loader_test = DataLoader(
        drug2_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None
    )

    model = modeling().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_AUCs_test = "fold_" + str(i) + ".csv"
    file_AUCs_test = os.path.join(RESULTS_DIR, file_AUCs_test)
    AUCs = "Epoch,AUC_dev,PR_AUC,ACC,BACC,PREC,TPR,KAPPA,RECALL,F1"
    with open(file_AUCs_test, "w") as f:
        f.write(AUCs + "\n")

    best_auc_train = 0
    best_auc_test = 0
    for epoch in range(NUM_EPOCHS):
        train(
            model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1
        )
        T_p, S_p, Y_p = predicting(model, device, drug1_loader_test, drug2_loader_test)
        # T is correct label
        # S is predict score
        # Y is predict label
        best_auc_test, auc_test = compute_preformence(
            T_p, S_p, Y_p, best_auc_test, file_AUCs_test
        )
