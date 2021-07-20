# Modified from GCNG, Get data from seqfish_plus
import pandas as pd
import numpy as np
import seaborn as sns
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import scipy.sparse
import pickle

####################  get the whole dataset
current_path = os.path.abspath('.')
cortex_svz_cellcentroids = pd.read_csv(current_path+'/seqfish_plus/cortex_svz_cellcentroids.csv')
############# get batch adjacent matrix
cell_view_list = []
for view_num in range(7):
    cell_view = cortex_svz_cellcentroids[cortex_svz_cellcentroids['Field of View']==view_num]
    cell_view_list.append(cell_view)

coords_list = []
print ('Add coordinates')
count = 0
for view_num in range(7):
    print (view_num)
    cell_view = cell_view_list[view_num]
    for i in range(cell_view.shape[0]):
        coords_list.append((cell_view.iloc[i]['X'],cell_view.iloc[i]['Y']))
        count += 1


np.save(current_path+'/seqfish_plus/coords_array.npy',np.array(coords_list))


# USE GCNG normalization
cortex_svz_counts = pd.read_csv(current_path+'/seqfish_plus/cortex_svz_counts.csv')
cortex_svz_counts_N =cortex_svz_counts.div(cortex_svz_counts.sum(axis=1)+1, axis='rows')*10**4
cortex_svz_counts_N.columns =[i.lower() for i in list(cortex_svz_counts_N)] ## gene expression normalization
cortex_svz_counts_N = cortex_svz_counts_N.T
cortex_svz_counts_N.to_csv(current_path+'/seqfish_plus/Use_expression.csv')



# USE scGNN style to only use top 2000 genes
# pd = pd.read_csv(current_path+'/seqfish_plus_2000/cortex_svz_counts.csv')
# pd1 = pd.T
# pd1.to_csv(current_path+'/seqfish_plus_2000/using.csv')
# Then run using PreprocessingscGNN.py
# python3 -W ignore PreprocessingscGNN.py --datasetName using.csv --datasetDir seqfish_plus_2000/ --LTMGDir seqfish_plus_2000/ --filetype CSV --geneRatio 1.00 --geneSelectnum 2000

# Step 2:
# Graph from scGNN to GCNG
def convert_adj_matrix(graphFilename,nodesize):
    row=[]
    col=[]
    data=[]
    with open(graphFilename) as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            if count>0:
                line = line.strip()
                words = line.split(',')
                row.append(words[0])
                col.append(words[1])
                data.append(int(float(words[2])))
            count += 1
        f.close()
    row = np.asarray(row)
    col = np.asarray(col)
    data = np.asarray(data)
    mtx = scipy.sparse.csc_matrix((data, (row, col)), shape=(nodesize, nodesize))
    mtx = mtx.todense()
    return mtx

graphFilename = '/Users/juexinwang/workspace/scGNNsp/outputdir/seqfish_plus_graph.csv'
mtx = convert_adj_matrix(graphFilename,913)
distance_matrix_threshold_I = np.float32(mtx)

#not normalize 
distance_matrix_threshold_I_crs = scipy.sparse.csr_matrix(distance_matrix_threshold_I)
with open(current_path+'/seqfish_plus/scGNNdist', 'wb') as fp:
    pickle.dump(distance_matrix_threshold_I_crs, fp)

#normalize
# distance_matrix_threshold_I_N = spektral.utils.normalized_adjacency(whole_distance_matrix_threshold_I, symmetric=True)
# distance_matrix_threshold_I_N = np.float32(whole_distance_matrix_threshold_I)
# distance_matrix_threshold_I_N_crs = sparse.csr_matrix(distance_matrix_threshold_I_N)
# with open(current_path+'/seqfish_plus/whole_FOV_distance_I_N_crs_'+str(threshold), 'wb') as fp:
#     pickle.dump(distance_matrix_threshold_I_N_crs, fp)


       
