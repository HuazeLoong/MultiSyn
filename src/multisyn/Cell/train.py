import argparse
import os
import random
import logging

from datetime import datetime
from time import perf_counter as t

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl

from torch.utils.data import DataLoader

from const import *
from Cell.model import GATEncoder, Cell2Vec
from dataset import C2VDataset
from utils import save_best_model, save_and_visual_loss, find_best_model

def train_step(mdl: Cell2Vec, g: dgl.DGLGraph, node_x, node_idx, cell_idx, y_true):
    """Training function: forward propagation, back propagation, update model parameters"""
    mdl.train()
    optimizer.zero_grad()
    y_pred = mdl(g, node_x, node_idx, cell_idx)
    step_loss = loss_func(y_pred, y_true)
    step_loss.backward()
    optimizer.step()

    return step_loss.item()


def gen_emb(mdl: Cell2Vec):
    """Get the embedding matrix of the model"""
    mdl.eval()
    with torch.no_grad():
        emb = mdl.embeddings.weight.data
    return emb.cpu().numpy()


def p_type(x):
    """Ensure that floating point numbers are in the range [0, 1)"""
    if isinstance(x, list):
        for xx in x:
            assert 0 <= xx < 1
    else:
        assert 0 <= x < 1
    return x


def get_graph_data():
    """Load the preprocessed PPI edge file and node feature file to build the graph structure input"""
    edges = np.load(COO_FILE).astype(int).transpose()
    eid = torch.from_numpy(edges)
    feat = np.load(NODE_FEAT_FILE)
    feat = torch.from_numpy(feat).float()
    return eid, feat


if __name__ == '__main__':
    gedatafile = 'ge'
    mutdatafile = 'mut'
    hidden_dim = 128
    emb_dim = 384
    suffix = 'sample'
    LR = 1e-3
    conv = 2
    batch = 2
    epoch = 3000
    gpu = None
    keep = 1
    patience = 50

    # ---------- Iterating over GE and MUT data types ---------- #
    for datafile in [gedatafile, mutdatafile]:
        t_type = datafile
        mdl_dir = os.path.join(DATA_DIR, 'mdl_{}_{}x{}_{}'.format(t_type, hidden_dim, emb_dim, suffix))
        loss_file = os.path.join(mdl_dir, 'loss.txt')
        if not os.path.exists(mdl_dir):
            os.makedirs(mdl_dir)

        torch.manual_seed(23333)
        random.seed(12345)

        learning_rate = LR
        num_layers = conv
        batch_size = batch
        num_epochs = epoch
        weight_decay = 1e-5

        edge_indices, node_features = get_graph_data()
        graph = dgl.graph((edge_indices[0], edge_indices[1]), idtype=torch.int32)
        graph = dgl.add_self_loop(graph)
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print('The code uses GPU...')
        else:
            device = torch.device('CPU')
            print('The code uses CPU!!!')
        graph = graph.to(device)
        node_features = node_features.to(device)

        if datafile == 'ge':
            c2v_dataset = C2VDataset(TARGET_GE, NODES_GE)
        else:
            print('datafile', datafile)
            c2v_dataset = C2VDataset(TARGET_MUT, NODES_MUT)

        dataloader = DataLoader(c2v_dataset, shuffle=True, num_workers=2)
        node_indices = c2v_dataset.node_indices.to(device)

        num_heads = 8 
        encoder = GATEncoder(
            in_features = node_features.shape[1], 
            out_features = hidden_dim, 
            num_heads = num_heads, 
            activation = F.relu, 
            k = num_layers
        ).to(device)
        
        # encoder = GCNEncoder(node_features.shape[1], hidden_dim, activation, k=num_layers).to(device)
        
        model = Cell2Vec(encoder, len(c2v_dataset), emb_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_func = nn.MSELoss(reduction='mean')

        print("Check model.")
        print(model)

        print("Start training.")
        losses = []
        min_loss = float('inf')
        angry = 0

        start = t()
        prev = start
        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0
            now = t()
            for step, batch in enumerate(dataloader):
                batch_x, batch_y = batch
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                loss = train_step(model, graph, node_features, node_indices, batch_x, batch_y)
                epoch_loss += loss * len(batch_x)
            epoch_loss /= len(c2v_dataset)
            logging.info('Epoch={:04d} Loss={:.4f}'.format(epoch, epoch_loss))
            print('Epoch={:04d} Loss={:.4f}'.format(epoch, epoch_loss))
            losses.append(epoch_loss)
            prev = now
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                save_best_model(model.state_dict(), mdl_dir, epoch, keep)
                angry = 0
            else:
                angry += 1
            # if angry == patience:
                # break

        print("Training completed.")
        print("Min train loss: {:.4f} | Epoch: {:04d}".format(min_loss, losses.index(min_loss)))
        print("Save to {}".format(mdl_dir))

        save_and_visual_loss(losses, loss_file, title='Train Loss', xlabel='epoch', ylabel='Loss')
        print("Save train loss curve to {}".format(loss_file))

        model.load_state_dict(torch.load(find_best_model(mdl_dir), map_location=torch.device('cpu')))
        embeddings = gen_emb(model)
        np.save(os.path.join(mdl_dir, 'embeddings.npy'), embeddings)
        print("Save {}".format(os.path.join(mdl_dir, 'embeddings.npy')))
