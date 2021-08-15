
# import warnings

# import faiss
# import torch

# from .torch_utils import to_numpy, to_torch
# from torch.autograd import Variable
# # __all__ = ["label_generator_kmeans"]
# #
# import numpy as np
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# @torch.no_grad()
# def label_generator_kmeans(features, num_classes=8, cuda=True):

#     # assert cfg.TRAIN.PSEUDO_LABELS.cluster == "kmeans"
#     # assert num_classes, "num_classes for kmeans is null"

#     # num_classes = cfg.TRAIN.PSEUDO_LABELS.cluster_num

#     # if not cfg.TRAIN.PSEUDO_LABELS.use_outliers:
#     #     warnings.warn("there exists no outlier point by kmeans clustering")

#     # k-means cluster by faiss
#     feat = to_numpy(features)
#     output=[]
#     for i in range(2):
#         # print(feat[i].shape)
#         f = feat[i].reshape(-1,feat[i].shape[0])
#         # print(f.shape)


#         cluster = faiss.Kmeans(
#             to_torch(f).size(-1), num_classes, niter=300, verbose=True, gpu=cuda
#         )
#         # print(features.shape)
#         # print(to_numpy(features))

#         # cluster.train(to_numpy(features))
#         cluster.train(f)
#         centers = to_torch(cluster.centroids).float()
#         _, labels = cluster.index.search(f, 1)
#         labels = labels.reshape(-1)
#         labels = to_torch(labels).long()
#         # print('labels',labels.shape)
#         # print('center',centers.shape)
#         onehot_label = dense_to_one_hot(to_numpy(labels), num_classes=8)
#         # print(onehot_label)
#         # print(onehot_label.shape)
#         if i==0:
#             output_0= onehot_label.reshape(onehot_label.shape[1], 512,512)
#         if i==1:
#             output_1 = onehot_label.reshape(onehot_label.shape[1], 512, 512)

#     # k-means does not have outlier points

#         assert not (-1 in labels)
#     output_final = torch.stack([to_torch(output_0),to_torch(output_1)],dim=0)
#     # print(output_final.shape)
#     return output_final.to(device)

# def label_generator_kmeans_test(features, num_classes=8, cuda=True):

#     # assert cfg.TRAIN.PSEUDO_LABELS.cluster == "kmeans"
#     # assert num_classes, "num_classes for kmeans is null"

#     # num_classes = cfg.TRAIN.PSEUDO_LABELS.cluster_num

#     # if not cfg.TRAIN.PSEUDO_LABELS.use_outliers:
#     #     warnings.warn("there exists no outlier point by kmeans clustering")

#     # k-means cluster by faiss
#     feat = to_numpy(features)
#     output=[]
#     for i in range(1):
#         # print(feat[i].shape)
#         f = feat[i].reshape(-1,feat[i].shape[0])
#         # print(f.shape)


#         cluster = faiss.Kmeans(
#             to_torch(f).size(-1), num_classes, niter=300, verbose=True, gpu=cuda
#         )
#         # print(features.shape)
#         # print(to_numpy(features))

#         # cluster.train(to_numpy(features))
#         cluster.train(f)
#         centers = to_torch(cluster.centroids).float()
#         _, labels = cluster.index.search(f, 1)
#         labels = labels.reshape(-1)
#         labels = to_torch(labels).long()
#         # print('labels',labels.shape)
#         # print('center',centers.shape)
#         onehot_label = dense_to_one_hot(to_numpy(labels), num_classes=8)
#         # print(onehot_label)
#         # print(onehot_label.shape)
#         if i==0:
#             output_0= onehot_label.reshape(1,onehot_label.shape[1], 256,256)
#         # if i==1:
#         #     output_1 = onehot_label.reshape(onehot_label.shape[1], 16, 16)

#     # k-means does not have outlier points

#         assert not (-1 in labels)
#     output_final = to_torch(output_0)
#     # print(output_final.shape)
#     return output_final.to(device)


# def dense_to_one_hot(labels_dense, num_classes):
#     """Convert class labels from scalars to one-hot vectors."""
#     num_labels = labels_dense.shape[0]
#     index_offset = np.arange(num_labels) * num_classes
#     labels_one_hot = np.zeros((num_labels, num_classes))
#     labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
#     return labels_one_hot
