import random, torch
import pandas as pd
import os,csv,re,json
import shutil
import anndata
import numpy as np
import cv2
import scanpy as sc
import scipy.sparse as sp
import torch.nn.functional as F
import anndata
import torch
import time
import datetime
from scipy.spatial import distance_matrix, minkowski_distance, distance
import networkx as nx
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score
def load_data(h5_path, spatial_path, scale_factor_path):
    # Read in gene expression and spatial location
    adata = sc.read_10x_h5(h5_path)
    spatial_all = pd.read_csv(spatial_path, sep=",", header=None, na_filter=False, index_col=0)
    spatial = spatial_all[spatial_all[1] == 1]
    spatial = spatial.sort_values(by=0)
    assert all(adata.obs.index == spatial.index)
    adata.obs["in_tissue"] = spatial[1]
    adata.obs["array_row"] = spatial[2]
    adata.obs["array_col"] = spatial[3]
    adata.obs["pxl_col_in_fullres"] = spatial[4]
    adata.obs["pxl_row_in_fullres"] = spatial[5]
    adata.obs.index.name = 'barcode'
    adata.var_names_make_unique()
    # Read scale_factor_file
    with open(scale_factor_path) as fp_scaler:
        scaler = json.load(fp_scaler)
    adata.uns["spot_diameter_fullres"] = scaler["spot_diameter_fullres"]
    adata.uns["tissue_hires_scalef"] = scaler["tissue_hires_scalef"]
    adata.uns["fiducial_diameter_fullres"] = scaler["fiducial_diameter_fullres"]
    adata.uns["tissue_lowres_scalef"] = scaler["tissue_lowres_scalef"]
    return adata, spatial_all

def readSpatial(filename):
    """
    Read spatial information
    """
    spatialMatrix = np.load(filename)
    return spatialMatrix

def preprocessSpatial(originalMatrix):
    """
    Preprocess spatial information
    Only works for 2D now, can be convert to 3D if needed
    Normalize all the coordinates to [-0.5,0.5]
    center is [0., 0.]
    D is maximum value in the x/y dim
    """
    spatialMatrix = np.zeros((originalMatrix.shape[0],originalMatrix.shape[1]))
    x = originalMatrix[:, 0]
    y = originalMatrix[:, 1]
    rangex = max(x)-min(x)
    rangey = max(y)-min(y)
    spatialMatrix[:, 0] = (x-min(x))/rangex-0.5
    spatialMatrix[:, 1] = (y-min(y))/rangey-0.5
    return spatialMatrix

def calculateSpatialMatrix(featureMatrix, distanceType='euclidean', k=6, pruneTag='NA', spatialMatrix=None):
    r"""
    Calculate spatial Matrix directly use X/Y coordinates
    """
    edgeList=[]

    ## Version 2: for each of the cell, calculate dist, save memory
    p_time = time.time()
    for i in np.arange(spatialMatrix.shape[0]):
        if i%10000==0:
            print('Start pruning '+str(i)+'th cell, cost '+str(time.time()-p_time)+'s')
        tmp=spatialMatrix[i,:].reshape(1,-1)
        distMat = distance.cdist(tmp,spatialMatrix, distanceType)
        # if k == 0, then use all the possible data
        # minus 1 for not exceed the array size
        if k == 0:
            k = spatialMatrix.shape[0]-1
        res = distMat.argsort()[:k+1]
        tmpdist = distMat[0,res[0][1:k+1]]
        boundary = np.mean(tmpdist)+np.std(tmpdist) #optional
        for j in np.arange(1,k+1):
            # No prune
            edgeList.append((i,res[0][j],1.0))
    return edgeList

def edgeList2edgeDict(edgeList, nodesize):
    graphdict={}
    tdict={}

    for edge in edgeList:
        end1 = edge[0]
        end2 = edge[1]
        tdict[end1]=""
        tdict[end2]=""
        if end1 in graphdict:
            tmplist = graphdict[end1]
        else:
            tmplist = []
        tmplist.append(end2)
        graphdict[end1]= tmplist

    #check and get full matrix
    for i in range(nodesize):
        if i not in tdict:
            graphdict[i]=[]

    return graphdict
def coords_to_adj(zOut, coords):
    SpatialMatrix = preprocessSpatial(coords)
    SpatialMatrix = torch.from_numpy(SpatialMatrix)
    SpatialMatrix = SpatialMatrix.type(torch.FloatTensor)
    prunetype = 'spatialGrid'
    knn_distance = 'euclidean'
    k = 6
    pruneTag = 'NA'
    useGAEembedding = True
    adj, edgeList = generateAdj(zOut, graphType=prunetype, para=knn_distance + ':' + str(k) + ':' +pruneTag, adjTag=useGAEembedding, spatialMatrix=SpatialMatrix)
    return adj, edgeList
def generateAdj(featureMatrix, graphType='spatialGrid', para=None, parallelLimit=0, adjTag=True, spatialMatrix=None):
    """
    Generating edgeList
    """
    edgeList = None
    adj = None

    if graphType == 'spatialGrid':
        # spatial Grid X,Y as the edge
        if para != None:
            parawords = para.split(':')
            distanceType = parawords[0]
            k = int(parawords[1])
            pruneTag = parawords[2]
        edgeList = calculateSpatialMatrix(featureMatrix, distanceType=distanceType, k=k, pruneTag=pruneTag, spatialMatrix=spatialMatrix)
    else:
        print('Should give graphtype')

    if adjTag:
        graphdict = edgeList2edgeDict(edgeList, featureMatrix.shape[0])
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))
    return adj, edgeList


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = np.arange(edges.shape[0])
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        if ~ismember([idx_i,idx_j],edges_all) and ~ismember([idx_j,idx_i],edges_all):
            val_edges_false.append([idx_i, idx_j])
        else:
            # Debug
            print(str(idx_i)+" "+str(idx_j))
        # Original:
        # val_edges_false.append([idx_i, idx_j])

    #TODO: temporary disable for ismember function may require huge memory.
    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def sparse_to_COO(adj):
    tmp_coo = sp.coo_matrix(adj)
    values = tmp_coo.data
    indices = np.vstack((tmp_coo.row,tmp_coo.col))
    i = torch.LongTensor(indices)
    # print('indices',indices)
    # print('i',i)
    # v = torch.LongTensor(values)
    # edge_index = torch.sparse_coo_tensor(i,v,tmp_coo.shape)
    return i
    # return edge_index

def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score
def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=labels * pos_weight)

    # Check if the model is simple Graph Auto-encoder
    if logvar is None:
        return cost

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

def loss_function_GVAE(preds, labels, mu, logstd, norm, pos_weight):
    # print(preds.size(),labels.size())   [21908]   [4226,4226]
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=labels * pos_weight)

    KLD = -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))
    return cost + KLD

def earlystopping(new_loss, best_loss, es):
    if new_loss<best_loss:
        best_loss = new_loss
        # print('------------new_loss---------------', new_loss)
        # print('------------best_loss---------------', best_loss)
        es = 0
    else:
        es = es + 1
    # print('--------------best_loss---------------', best_loss)
    # print('--------------es---------------', es)
    return best_loss, es

def coords_prepare(spatialMatrix):
    spatialMatrix = preprocessSpatial(spatialMatrix)
    spatialMatrix = torch.from_numpy(spatialMatrix)
    spatialMatrix = spatialMatrix.type(torch.FloatTensor)
    return spatialMatrix
