import os, sys
# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
# For replicating the experiments
import argparse
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
from GAEdata_prepare import *
from GAEgraph_prepare import *
from util_function import loss_function_graph
from torch_geometric.data import Data
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from sklearn.decomposition import PCA

def GAEembedding(z, adj, GAEmodel, scgnnsp_alpha, scgnnsp_zdim, enc_type, coords):
    '''
    GAE embedding for clustering
    Param:
        z,adj
    Return:
        Embedding from graph
    '''
    # featrues from z
    # Louvain
    torch.manual_seed(1)
    features = z

    coords = coords_prepare(coords)
    features = torch.FloatTensor(features)
    n_nodes, feat_dim = features.shape

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    edge_index = sparse_to_COO(adj)

    data = Data(x=features, edge_index=edge_index)

    # Some preprocessing
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    GAElr = 0.001
    GAEepochs = 500
    #-----------------GAEmodel-------------#
    if GAEmodel == 'vgae':
        model = VGAE(VGAEencoder(feat_dim, int(scgnnsp_zdim),float(scgnnsp_alpha),enc_type),InnerProductDecoder(), feat_dim, int(scgnnsp_zdim))
    else:
        model = GAE(GAEencoder(feat_dim),InnerProductDecoder())

    optimizer = optim.Adam(model.parameters(), lr=GAElr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, verbose=True)
    # print("初始化的学习率：", optimizer.defaults['lr'])
    device = torch.device('cpu')
    model = model.to(device)
    data = data.to(device)
    adj_label = adj_label.to(device)
    best_loss = 1000
    es = 0
    patience = 15
    x = data.x
    edge_index = data.edge_index

    for epoch in tqdm(range(GAEepochs)):
        t = time.time()

        model.train()
        optimizer.zero_grad()
        z, mu, logstd, po_emb = model.encode(x, edge_index,coords)
        gae_preds = model.decoder(z)
        recon_batch = model.feature_decoder(po_emb)

        mu_dummy = ''
        logvar_dummy = ''
        gammaPara = 0.1
        regulationMatrixBatch = None
        regulized_type = 'noregu'
        alphaRegularizePara = 0.9
        model_mode = 'PAE'
        reduction = 'sum'

        gae_loss = loss_function(preds=gae_preds, labels=adj_label, mu=mu, logvar=logstd, n_nodes=n_nodes, norm=norm, pos_weight=pos_weight)
        fae_loss = loss_function_graph(recon_batch, x.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy,
                                   gammaPara=gammaPara,
                                   regulationMatrix=regulationMatrixBatch, regularizer_type=regulized_type,
                                   reguPara=alphaRegularizePara, modelusage=model_mode, reduction=reduction)

        ###earlystopping
        # best_loss, es = earlystopping(new_loss=loss, best_loss=best_loss, es=es)
        # if es > patience:
        #     print("Early stopping with best_loss: ", best_loss)
        #     break
        loss = gae_loss*10 + fae_loss*0.01
        loss.backward()
        cur_loss = loss.item()
        print(str(epoch)+': loss----------->',cur_loss)
        optimizer.step()
        # print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        scheduler.step(loss)

    z = val_test(model, x, edge_index,coords)
    hidden_emb = z#.data.cpu().numpy()
    tqdm.write("Optimization Finished!")
    return hidden_emb

def val_test(model, x, edge_index,coords):
    model.eval()
    z, mu, logstd, po_emb = model.encode(x, edge_index,coords)
    return z
