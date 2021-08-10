# Written by Yixiao Ge

import warnings
import cv2
import faiss
import torch
from sklearn.cluster import KMeans
from ..torch_utils import to_numpy, to_torch
from torch.autograd import Variable
# __all__ = ["label_generator_kmeans"]
#
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@torch.no_grad()

def kmeans(seg_logit, n):

    clf = KMeans(n_clusters=n, random_state=5)
    y_pred = clf.fit_predict(to_numpy(seg_logit[0].reshape(seg_logit.shape[1],-1).t()))
    y_pred  = y_pred .reshape(1,1,seg_logit.shape[2],seg_logit.shape[3])
    print(clf.cluster_centers_)
    np.save('label',y_pred[0][0])
    print(y_pred.shape)
    return to_torch(y_pred).to(device)


def kmeans_debackground(img_meta, seg_logit, n):
    seg_logit = to_numpy(seg_logit)
    sample = img_meta[0]['ori_filename'].split('_',5)[0]
    label = cv2.imread(img_meta[0]['filename'], cv2.IMREAD_GRAYSCALE)
    x,y = np.where(label==255)
    for j in range(len(x)):
        seg_logit[0,:,x[j],y[j]] = [ -1000,-1000, -1000, -1000, -1000, -1000, -1000, -1000]
        # print(seg_logit[0,:,x[j],y[j]])
    clf = KMeans(n_clusters=n, random_state=5)
    y_pred = clf.fit_predict(seg_logit[0].reshape(seg_logit.shape[1],-1).T)
    y_pred  = y_pred .reshape(1,1,seg_logit.shape[2],seg_logit.shape[3])
    a,b,c,d = np.where(y_pred==0)  
    backgroud = y_pred[0,0,0,0]
    for k in range(len(c)):
        y_pred[0,0,c[k],d[k]] = backgroud
    for k in range(len(x)):
        y_pred[0,0,x[k],y[k]] = 0
    return to_torch(y_pred).to(device)