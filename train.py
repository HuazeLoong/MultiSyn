import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from random import shuffle
from model import *
from torch.optim import SGD
from sklearn import metrics
from dataset_drug import MyTestDataset
from torch_geometric.loader import DataLoader 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix,f1_score,recall_score, balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score

def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch,loss_file):
    print('Training on {} samples...'.format(len(drug1_loader_train.dataset)))
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

        ys = F.softmax(output, 1).to('cpu').data.numpy()
        predicted_labels = list(map(lambda x: np.argmax(x), ys))
        predicted_scores = list(map(lambda x: x[1], ys))
        total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
        total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
        total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)

        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data1.y),
                                                                           len(drug1_loader_train.dataset),
                                                                           100. * batch_idx / len(drug1_loader_train),
                                                                           loss.item()))

    total_labels = total_labels.numpy().flatten()
    total_preds = total_preds.numpy().flatten()
    total_prelabels = total_prelabels.numpy().flatten()

    train_auc = roc_auc_score(total_labels, total_preds)
    print(f'Epoch: {epoch}, Train AUC: {train_auc:.4f}')
    
    avg_loss = total_loss / len(drug1_loader_train)
    with open(loss_file, 'a') as f:
        f.write(f'{epoch},train,{avg_loss:.6f}\n')

    return total_labels, total_preds, total_prelabels


def predicting(model, device, drug1_loader_test, drug2_loader_test):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()

    print('Make prediction for {} samples...'.format(len(drug1_loader_test.dataset)))
    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1 = data[0]
            data2 = data[1]
            data1 = data1.to(device)
            data2 = data2.to(device)
            output = model(data1, data2)
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()

def compute_preformence(T,S,Y,best_auc,file,tag):
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

    AUCs = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, recall,F1]
    if best_auc < AUC:
        save_AUCs(AUCs, file)
        best_auc = AUC
    return best_auc,AUC


modeling = TestSyn

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1500

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
datafile = 'new_labels_0_10'
cellfile = 'data/new_cell_features_954.csv'

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('The code uses GPU...')
else:
    device = torch.device('CPU')
    print('The code uses CPU!!!')

drug1_data = MyTestDataset(root='TEST\data', dataset=datafile + '_drug1')
drug2_data = MyTestDataset(root='TEST\data', dataset=datafile + '_drug2')
lenth = len(drug1_data)
pot = int(lenth/5)
print('lenth', lenth)
print('pot', pot)

random_num = random.sample(range(0, lenth), lenth)
for i in range(5):
    test_num = random_num[pot*i:pot*(i+1)]
    train_num = random_num[:pot*i] + random_num[pot*(i+1):]

    drug1_data_train = drug1_data[train_num]
    drug1_data_test = drug1_data[test_num]
    drug1_loader_train = DataLoader(drug1_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
    drug1_loader_test = DataLoader(drug1_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None)

    drug2_data_test = drug2_data[test_num]
    drug2_data_train = drug2_data[train_num]
    drug2_loader_train = DataLoader(drug2_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
    drug2_loader_test = DataLoader(drug2_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None)

    model = modeling().to(device)
    loss_fn = nn.CrossEntropyLoss()    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    os.makedirs('TEST/data/result', exist_ok=True)
    os.makedirs('TEST/data/result/results', exist_ok=True)
    os.makedirs('TEST/data/result/loss', exist_ok=True)
    file_AUCs_train = 'TEST/data/result/results/train' + str(i) + '.csv'
    file_AUCs_test = 'TEST/data/result/results/test' + str(i) + '.csv'
    loss_file = 'TEST/data/result/loss/loss' + str(i)  + '.csv'
    AUCs = ('Epoch,AUC_dev,PR_AUC,ACC,BACC,PREC,TPR,KAPPA,RECALL,F1')
    for file_AUCs in [file_AUCs_train, file_AUCs_test]:
        with open(file_AUCs, 'w') as f:
            f.write(AUCs + '\n')

    best_auc_train = 0
    best_auc_test = 0
    for epoch in range(NUM_EPOCHS):
        T_t, S_t, Y_t = train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1,loss_file)
        best_auc_train,auc_train = compute_preformence(T_t, S_t, Y_t,best_auc_train,file_AUCs_train,0)
        T_p, S_p, Y_p = predicting(model, device, drug1_loader_test, drug2_loader_test)
        # T is correct label
        # S is predict score
        # Y is predict label
        best_auc_test,auc_test = compute_preformence(T_p, S_p, Y_p,best_auc_test,file_AUCs_test,1)
        # scheduler.step(auc_test)
