from skimage import io
# import SpaGCN as spg
from SpaGCN2 import SpaGCN
import cv2
import numpy as np
from sklearn.decomposition import PCA
from SpaGCN2.calculate_adj import calculate_adj_matrix
import random, torch

def generate_embedding(anndata, pca, res,img_path,pca_opt,method='spaGCN'):
    if method == 'spaGCN':
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
        clf.train(anndata, adj ,num_pcs=pca, init_spa=True, init="louvain", res=res, tol=5e-3,pca_opt = pca_opt)
        y_pred, prob, z = clf.predict_with_embed()
        return z
    # elif method == 'scGNN':