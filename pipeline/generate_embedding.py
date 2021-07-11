from skimage import io
# import SpaGCN as spg
from SpaGCN2 import SpaGCN
import cv2
import numpy as np
from sklearn.decomposition import PCA
from SpaGCN2.calculate_adj import calculate_adj_matrix
import random, torch
import pandas as pd
import os
import shutil

# def generate_embedding(anndata, pca, res,img_path,pca_opt,method='spaGCN'):
#     if method == 'spaGCN':
#         # image = io.imread("img/"+sample+".png")  # value
#         # Calculate adjacent matrix
#         # b = 49
#
#         random.seed(200)
#         torch.manual_seed(200)
#         np.random.seed(200)
#
#         b = 49
#         a = 1
#         x2 = anndata.obs["array_row"].tolist()
#         x3 = anndata.obs["array_col"].tolist()
#         x4 = anndata.obs["pxl_col_in_fullres"]
#         x5 = anndata.obs["pxl_row_in_fullres"]
#         if img_path != None:
#             image = io.imread(img_path)
#             # print(image)
#             max_row = max_col = int((2000 / anndata.uns['tissue_hires_scalef']) + 1)
#             x4 = x4.values * (image.shape[0] / max_row)
#             x4 = x4.astype(np.int)
#             x4 = x4.tolist()
#             x5 = x5.values * (image.shape[1] / max_col)
#             x5 = x5.astype(np.int)
#             x5 = x5.tolist()
#             adj = calculate_adj_matrix(x=x2, y=x3, x_pixel=x4, y_pixel=x5, image=image, beta=b, alpha=a,
#                                        histology=True)  # histology optional
#         else:
#             x4 = x4.tolist()
#             x5 = x5.tolist()
#             adj = calculate_adj_matrix(x=x2, y=x3, x_pixel=x4, y_pixel=x5, beta=b, alpha=a,
#                                           histology=False)
#         # print(adj[2000].size)
#         # print(adj.shape)
#         p = 0.5
#         # l = spg.find_l(p=p, adj=adj, start=0.75, end=0.8, sep=0.001, tol=0.01)
#         l = 1.43
#         # res = 0.6
#         clf = SpaGCN()
#         clf.set_l(l)
#         # Init using louvain
#         # clf.train(anndata, adj,num_pcs=pca, init_spa=True, init="louvain",louvain_seed=0, res=res, tol=5e-3)
#         clf.train(anndata, adj ,num_pcs=pca, init_spa=True, init="louvain", res=res, tol=5e-3,pca_opt = pca_opt)
#         y_pred, prob, z = clf.predict_with_embed()
#         return z
#     # elif method == 'scGNN':

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