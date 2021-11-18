from skimage import io
# import SpaGCN as spg
from SpaGCN2 import SpaGCN
import cv2
import numpy as np
from sklearn.decomposition import PCA
from SpaGCN2.calculate_adj import calculate_adj_matrix
import argparse
import scanpy as sc
from src.graph_func import graph_construction
from src.SEDR_train import SEDR_Train
import random, torch
import pandas as pd
import os
import shutil
import anndata



def generate_embedding_sp(anndata, pca, res, img_path,pca_opt):
    # image = io.imread("img/"+sample+".png")  # value
    # Calculate adjacent matrix
    # b = 49

    random.seed(200)
    torch.manual_seed(200)
    np.random.seed(200)

    b = 49
    a = 1
    x2 = anndata.obs["array_row"].tolist()
    x3 = anndata.obs["array_col"].tolist()
    x4 = anndata.obs["pxl_col_in_fullres"]
    x5 = anndata.obs["pxl_row_in_fullres"]
    if img_path != None:
        image = io.imread(img_path)
        # print(image)
        max_row = max_col = int((2000 / anndata.uns['tissue_hires_scalef']) + 1)
        x4 = x4.values * (image.shape[0] / max_row)
        x4 = x4.astype(np.int)
        x4 = x4.tolist()
        x5 = x5.values * (image.shape[1] / max_col)
        x5 = x5.astype(np.int)
        x5 = x5.tolist()
        adj = calculate_adj_matrix(x=x2, y=x3, x_pixel=x4, y_pixel=x5, image=image, beta=b, alpha=a,
                                   histology=True)  # histology optional
    else:
        x4 = x4.tolist()
        x5 = x5.tolist()
        adj = calculate_adj_matrix(x=x2, y=x3, x_pixel=x4, y_pixel=x5, beta=b, alpha=a,
                                   histology=False)
    # print(adj[2000].size)
    # print(adj.shape)
    p = 0.5
    # l = spg.find_l(p=p, adj=adj, start=0.75, end=0.8, sep=0.001, tol=0.01)
    l = 1.43
    # res = 0.6
    clf = SpaGCN()
    clf.set_l(l)
    # Init using louvain
    # clf.train(anndata, adj,num_pcs=pca, init_spa=True, init="louvain",louvain_seed=0, res=res, tol=5e-3)
    clf.train(anndata, adj, num_pcs=pca, init_spa=True, init="louvain", res=res, tol=5e-3,pca_opt = pca_opt)
    y_pred, prob, z = clf.predict_with_embed()
    return z


def generate_embedding_sc(anndata, sample, scgnnsp_dist, scgnnsp_alpha, scgnnsp_k, scgnnsp_zdim, scgnnsp_bypassAE):
    scGNNsp_folder = "scGNNsp_space/"
    if not os.path.exists(scGNNsp_folder):
        os.makedirs(scGNNsp_folder)
    datasetName = sample+'_'+scgnnsp_zdim+'_'+scgnnsp_alpha+'_'+scgnnsp_k+'_'+scgnnsp_dist+'_logcpm'
    scGNNsp_data_folder = scGNNsp_folder + datasetName + '/'
    if not os.path.exists(scGNNsp_data_folder):
        os.makedirs(scGNNsp_data_folder)
    coords_list = [list(t) for t in zip(anndata.obs["array_row"].tolist(), anndata.obs["array_col"].tolist())]
    if not os.path.exists(scGNNsp_data_folder + 'coords_array.npy'):
        np.save(scGNNsp_data_folder + 'coords_array.npy', np.array(coords_list))
    original_cpm_exp = anndata.X.A.T
    # original_cpm_exp = pd.read_csv('scGNNsp_space/151507_logcpm_test/151507_human_brain_ex.csv', index_col=0).values
    if not os.path.exists(scGNNsp_data_folder + sample + '_logcpm_expression.csv'):
        pd.DataFrame(original_cpm_exp).to_csv(scGNNsp_data_folder + sample + '_logcpm_expression.csv')
    os.chdir(scGNNsp_folder)
    command_preprocessing = 'python -W ignore PreprocessingscGNN.py --datasetName ' + sample + '_logcpm_expression.csv --datasetDir ' + datasetName + '/ --LTMGDir ' + datasetName + '/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None'
    if not os.path.exists(datasetName + '/Use_expression.csv'):
        os.system(command_preprocessing)
    # python -W ignore PreprocessingscGNN.py --datasetName 151507_human_brain_ex.csv --datasetDir 151507_velocity/ --LTMGDir 151507_velocity/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None
    scgnnsp_output_folder = 'outputdir-3S-' + datasetName + '_EM1_resolution0.3_' + scgnnsp_dist + '_dummy_add_PEalpha' + scgnnsp_alpha + '_k' + scgnnsp_k +'_zdim' + scgnnsp_zdim+ '_NA/'  
    scgnnsp_output_embedding_csv = datasetName + '_' + scgnnsp_k + '_' + scgnnsp_dist + '_NA_dummy_add_' + scgnnsp_alpha + '_intersect_160_GridEx19_embedding.csv'
    command_scgnnsp = 'python -W ignore scGNNsp.py --datasetName ' + datasetName + ' --datasetDir ./  --outputDir ' + scgnnsp_output_folder + ' --resolution 0.3 --nonsparseMode --EM-iteration 1 --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda --debugMode savePrune --saveinternal --GAEhidden2 3 --prunetype spatialGrid --PEtypeOp add --pe-type dummy'
    command_scgnnsp = command_scgnnsp + " --knn-distance " + scgnnsp_dist
    command_scgnnsp = command_scgnnsp + " --PEalpha " + scgnnsp_alpha
    command_scgnnsp = command_scgnnsp + " --k " + scgnnsp_k
    command_scgnnsp = command_scgnnsp + " --zdim " + scgnnsp_zdim
    if scgnnsp_bypassAE:
        scgnnsp_output_folder = 'outputdir-3S-' + datasetName + '_EM1_resolution0.3_' + scgnnsp_dist + '_dummy_add_PEalpha' + scgnnsp_alpha + '_k' + scgnnsp_k + '_NA_bypassAE/'
        command_scgnnsp = 'python -W ignore scGNNsp.py --datasetName ' + datasetName + ' --datasetDir ./  --outputDir ' + scgnnsp_output_folder + ' --resolution 0.3 --nonsparseMode --EM-iteration 1 --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda --debugMode savePrune --saveinternal --GAEhidden2 3 --prunetype spatialGrid --PEtypeOp add --pe-type dummy'
        command_scgnnsp = command_scgnnsp + " --knn-distance " + scgnnsp_dist
        command_scgnnsp = command_scgnnsp + " --PEalpha " + scgnnsp_alpha
        command_scgnnsp = command_scgnnsp + " --k " + scgnnsp_k
        command_scgnnsp = command_scgnnsp + " --zdim " + scgnnsp_zdim
        command_scgnnsp = command_scgnnsp + " --bypassAE"
    if not os.path.exists(scgnnsp_output_folder + scgnnsp_output_embedding_csv):
        os.system(command_scgnnsp)
    scgnnsp_output_embedding = pd.read_csv(scgnnsp_output_folder + scgnnsp_output_embedding_csv, index_col=0).values
    if os.path.exists(sample + '_' + scgnnsp_zdim + '_' + scgnnsp_alpha + '_' + scgnnsp_k + '_' + scgnnsp_dist + '_logcpm'):
        shutil.rmtree(sample + '_' + scgnnsp_zdim + '_' + scgnnsp_alpha + '_' + scgnnsp_k + '_' + scgnnsp_dist + '_logcpm')
    if os.path.exists(scgnnsp_output_folder):
        shutil.rmtree(scgnnsp_output_folder)
    os.chdir(os.path.dirname(os.getcwd()))
    return scgnnsp_output_embedding

def generate_embedding_SEDR(adata, k: 'int' = 10, cell_feat_dim: 'int' = 200,
                   gcn_w: 'float' = 0.1):
    '''
    SEDR can be used for embedding used both gene expression and spatial
    coordinate. SEDR_embedding contains two steps, that are, SEDR embedding,
    where each spot will get 28 dimentional representation, and UMAP embedding,
    where each spot will get 3 dimentional representation eventually. The
    embedding results could be found using adataobsm['X_SEDR_umap'].

    Parameters
    ----------
    adata : anndata
        adata should contains adata.obs[['pxl_col_in_fullres']] and
        adata.obs[[ 'pxl_row_in_fullres']] to obtain spatial information.
        In addition, adata.X should be normlized before running SEDR_embedding
    k : int, optional
        parameter k in spatial graph. The default is 10.
    cell_feat_dim : int, optional
        Dim of PCA. The default is 200.
    gcn_w : float, optional
        Weight of GCN loss. The default is 0.1.

    Returns
    -------
    adata : anndata
        The SEDR+UMAP embedding results could be found using adata.obsm['X_SEDR_umap']

    '''

    # Set needed parameters
    torch.cuda.cudnn_enabled = False
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=10, help='parameter k in spatial graph')
    parser.add_argument('--knn_distanceType', type=str, default='euclidean',
                        help='graph distance type: euclidean/cosine/correlation')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--cell_feat_dim', type=int, default=200, help='Dim of PCA')
    parser.add_argument('--feat_hidden1', type=int, default=100, help='Dim of DNN hidden 1-layer.')
    parser.add_argument('--feat_hidden2', type=int, default=20, help='Dim of DNN hidden 2-layer.')
    parser.add_argument('--gcn_hidden1', type=int, default=32, help='Dim of GCN hidden 1-layer.')
    parser.add_argument('--gcn_hidden2', type=int, default=8, help='Dim of GCN hidden 2-layer.')
    parser.add_argument('--p_drop', type=float, default=0.2, help='Dropout rate.')
    parser.add_argument('--using_dec', type=bool, default=True, help='Using DEC loss.')
    parser.add_argument('--using_mask', type=bool, default=False, help='Using mask for multi-dataset.')
    parser.add_argument('--feat_w', type=float, default=10, help='Weight of DNN loss.')
    parser.add_argument('--gcn_w', type=float, default=0.1, help='Weight of GCN loss.')
    parser.add_argument('--dec_kl_w', type=float, default=10, help='Weight of DEC loss.')
    parser.add_argument('--gcn_lr', type=float, default=0.01, help='Initial GNN learning rate.')
    parser.add_argument('--gcn_decay', type=float, default=0.01, help='Initial decay rate.')
    parser.add_argument('--dec_cluster_n', type=int, default=10, help='DEC cluster number.')
    parser.add_argument('--dec_interval', type=int, default=20, help='DEC interval nnumber.')
    parser.add_argument('--dec_tol', type=float, default=0.00, help='DEC tol.')
    # ______________ Eval clustering Setting _________
    parser.add_argument('--eval_resolution', type=int, default=1, help='Eval cluster number.')
    parser.add_argument('--eval_graph_n', type=int, default=20, help='Eval graph kN tol.')
    params = parser.parse_args()
    params.device = device
    # _____________ Set current function's parameters
    params.k = k
    params.cell_feat_dim = cell_feat_dim
    params.gcn_w = gcn_w

    # Construct spatial graph for GCN embedding
    adata.obsm['spatial'] = adata.obs[['pxl_col_in_fullres', 'pxl_row_in_fullres']].to_numpy()
    sc.pp.filter_genes(adata, min_cells=3)  # Filter genes
    adata_X = adata.X
    adata_X = sc.pp.scale(adata_X)  # Scale count matrix
    adata_X = sc.pp.pca(adata_X, n_comps=cell_feat_dim)  # PCA
    graph_dict = graph_construction(adata.obsm['spatial'], adata.shape[0], params)

    # SEDR process
    params.cell_num = adata.shape[0]  # the number of total cells/spots
    sedr_net = SEDR_Train(adata_X, graph_dict, params)  # Training process
    sedr_net.train_with_dec()
    sedr_feat, _, _, _ = sedr_net.process()  # sedr_feat is the SEDR embedding

    # UMAP
    sedr_feat_df = pd.DataFrame(sedr_feat, index=adata.obs.index)
    adata_u = anndata.AnnData(sedr_feat_df)
    sc.pp.neighbors(adata_u)  # Constrcut neighboring graph
    sc.tl.umap(adata_u, n_components=3)  # UMAP embedding

    # Add umap embedding results to adata object
    adata.obsm['X_SEDR_umap'] = adata_u.obsm['X_umap']

    return adata

def generate_embedding_UMAP(adata,pc_num,neighbor):

    pca_embedding = sc.pp.pca(adata.X,n_comps=pc_num)
    pca_embed_df = pd.DataFrame(pca_embedding, index=adata.obs.index)

    adata_pca = anndata.AnnData(pca_embed_df)
    sc.pp.neighbors(adata_pca,n_neighbors=neighbor)
    sc.tl.umap(adata_pca, n_components=3)
    adata.obsm['embedding'] = adata_pca.obsm['X_umap']
    return adata

