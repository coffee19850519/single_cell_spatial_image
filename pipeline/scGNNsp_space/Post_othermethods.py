# Test results on all possible clustering methods using clustering results
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, FeatureAgglomeration, OPTICS, MeanShift
from model import AE, VAE, PVAE, PAE
from util_function import *
from graph_function import *
from benchmark_util import *

import argparse
parser = argparse.ArgumentParser(description='Main entrance of scGNN')
parser.add_argument('--dataName', type=str, default='151507_cpm', help='dataName')
args = parser.parse_args()

def readGraph(name):
    edgeList = []
    count = 0
    with open(name) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            words = line.split(',')
            if count > 0:
                # edgeList.append((words[0],words[1],words[2]))
                # For some version, add forced type conversion
                edgeList.append((int(words[0]),int(words[1]),float(words[2])))
            count +=1
    return edgeList

def readpreprocess(dataName):
    spatialMatrix = readSpatial('/Users/wangjue/workspace/scGNNsp/'+dataName+'/coords_array.npy')
    spatialMatrix = preprocessSpatial(spatialMatrix)
    spatialMatrix = torch.from_numpy(spatialMatrix)
    spatialMatrix = spatialMatrix.type(torch.FloatTensor)

    # df = pd.read_csv('/Users/wangjue/workspace/scGNNsp/outputdirMVKMm1-'+dataName+'_cpm/'+dataName+'_cpm_10_euclidean_STD_dummy_add_0.5_embedding.csv')
    df = pd.read_csv('/Users/wangjue/workspace/scGNNsp/outputdirS-'+dataName+'_cpm_0.3/'+dataName+'_cpm_8_euclidean_Grid_dummy_add_0.5_embedding.csv')
    filter_col = [col for col in df if col.startswith('embedding') ]
    dfEX = df[filter_col]
    zOut = dfEX.to_numpy()

    # adjTarget, edgeList = generateAdj(zOut, graphType='spatialGrid', para='euclidean:8:Grid', adjTag=True, spatialMatrix = spatialMatrix)
    # adjSource, edgeList = generateAdj(zOut, graphType='KNNgraphStatsSingleThread', para='euclidean:10:STD', adjTag=True, spatialMatrix = None)
    
    edgeList = readGraph('/Users/wangjue/workspace/scGNNsp/outputdirS-'+dataName+'_cpm_0.3/'+dataName+'_cpm_8_euclidean_Grid_dummy_add_0.5_graph.csv')
    
    return zOut,edgeList

def readembedding(dataName,k,pe_type,skStr):
    df = pd.read_csv('/storage/htc/joshilab/wangjue/scGNNsp/outputdirH-'+dataName+'_0.3/'+dataName+'_'+k+'_euclidean_NA_'+pe_type+'_add_0.5_intersect_'+skStr+'_embedding.csv')
    # df = pd.read_csv('/Users/wangjue/workspace/scGNNsp/outputdirS-151671_cpm_0.3/'+dataName+'_cpm_8_euclidean_Grid_dummy_add_0.5_embedding.csv')    
    filter_col = [col for col in df if col.startswith('embedding') ]
    dfEX = df[filter_col]
    zOut = dfEX.to_numpy()
    edgeList = readGraph('/storage/htc/joshilab/wangjue/scGNNsp/outputdirH-'+dataName+'_0.3/'+dataName+'_'+k+'_euclidean_NA_'+pe_type+'_add_0.5_intersect_'+skStr+'_graph.csv')
    # edgeList = readGraph('/Users/wangjue/workspace/scGNNsp/outputdirS-151671_cpm_0.3/'+dataName+'_cpm_8_euclidean_Grid_dummy_add_0.5_graph.csv')    
    return zOut,edgeList

def clusteringMethod(zOut, edgeList, name, preK=5, resolution = 0.3):
    if name=='Louvain':
        listResult, size = generateLouvainCluster(edgeList)
        k = len(np.unique(listResult))
        print('Louvain\t'+str(k)+'\t', end='')
    elif name== 'LouvainK':
        listResult, size = generateLouvainCluster(edgeList)
        k = len(np.unique(listResult))
        # print('Louvain cluster: '+str(k))
        k = int(k*resolution) if int(k*resolution) >= 3 else 2
        # print('LouvainK\t'+str(k)+'\t', end='')
        clustering = KMeans(n_clusters=k, random_state=0).fit(zOut)
        listResult = clustering.predict(zOut)

        #Check criteria
        intraArr=clustering.transform(zOut)
        intraL1=np.sum(intraArr)
        intraL2=np.sum(intraArr**2)
        print(str(clustering.score(zOut))+'\t'+str(intraL1)+'\t'+str(intraL2))
    elif name == 'LouvainB':
        listResult, size = generateLouvainCluster(edgeList)
        k = len(np.unique(listResult))
        print('LouvainB\t'+str(k)+'\t', end='')
        k = int(k*resolution) if int(k*resolution) >= 3 else 2
        clustering = Birch(n_clusters=k).fit(zOut)
        listResult = clustering.predict(zOut)
    elif name == 'KMeans':        
        clustering = KMeans(n_clusters=preK,random_state=0).fit(zOut)
        listResult = clustering.predict(zOut)
        print('KMeans\t'+str(len(set(listResult)))+'\t', end='')
    elif name == 'SpectralClustering':
        clustering = SpectralClustering(n_clusters=preK, assign_labels="discretize", random_state=0).fit(zOut)
        listResult = clustering.labels_.tolist()
        print('SpectralClustering\t'+str(len(set(listResult)))+'\t', end='')
    elif name == 'AffinityPropagation':
        clustering = AffinityPropagation().fit(zOut)
        listResult = clustering.predict(zOut)
        print('AffinityPropagation\t'+str(len(set(listResult)))+'\t', end='')
    elif name == 'AgglomerativeClustering':
        clustering = AgglomerativeClustering().fit(zOut)
        listResult = clustering.labels_.tolist()
        print('Agglo\t'+str(len(set(listResult)))+'\t', end='')
    elif name == 'AgglomerativeClusteringK':
        clustering = AgglomerativeClustering(n_clusters=preK).fit(zOut)
        listResult = clustering.labels_.tolist()
        print('AggloK\t'+str(len(set(listResult)))+'\t', end='')
    elif name == 'Birch':
        clustering = Birch(n_clusters=preK).fit(zOut)
        listResult = clustering.predict(zOut)
        print('Birch\t'+str(len(set(listResult)))+'\t', end='')
    elif name == 'BirchN':
        clustering = Birch(n_clusters=None).fit(zOut)
        listResult = clustering.predict(zOut)
        print('BirchN\t'+str(len(set(listResult)))+'\t', end='')
    elif name == 'MeanShift':
        clustering = MeanShift().fit(zOut)
        listResult = clustering.predict(zOut)
        print('MeanShift\t'+str(len(set(listResult)))+'\t', end='')
    elif name == 'OPTICS':
        clustering = OPTICS(min_samples=3, min_cluster_size=3).fit(zOut)
        # clustering = OPTICS(min_samples=int(args.k/2), min_cluster_size=args.minMemberinCluster).fit(zOut)
        listResult = clustering.predict(zOut)
        print('OPTICS\t'+str(len(set(listResult)))+'\t', end='')
    elif name=='Leiden':
        listResult, size = generateLeidenCluster(edgeList)
        print('Leiden\t'+str(len(set(listResult)))+'\t', end='')
    return listResult


def plotMethod(zOut, edgeList, method):
    listResult = clusteringMethod(zOut, edgeList, method, preK=6, resolution = 0.3)
    listResult = pd.Series(listResult)
    color_labels = listResult.unique()
    # print(color_labels)
    # List of colors in the color palettes
    rgb_values = sns.color_palette("Set2", len(color_labels))
    # Map continents to the colors
    color_map = dict(zip(color_labels, rgb_values))
    # Finally use the mapped values
    plt.scatter(arr[:,0], arr[:,1], c=listResult.map(color_map), s=10)
    # plt.show()
    plt.savefig(str(dataName)+'_'+method+'.png')
    plt.close()

    labelname = '/Users/wangjue/workspace/scGNNsp/'+dataName+'/label.csv'
    df = pd.read_csv(labelname)
    listBench = df['layer'].to_numpy().tolist()
    ari, ami, nmi, cs, fms, vms, hs = measureClusteringTrueLabel(listBench, listResult)
    print(ari)

# dataName = '151671'

# Original
# arr = np.load('/Users/wangjue/workspace/scGNNsp/'+dataName+'/coords_array.npy')
# zOut,edgeList = readpreprocess(dataName)
# # plotMethod(zOut, edgeList, 'Louvain')
# # plotMethod(zOut, edgeList, 'LouvainK')
# # plotMethod(zOut, edgeList, 'LouvainB')
# plotMethod(zOut, edgeList, 'KMeans')
# # plotMethod(zOut, edgeList, 'SpectralClustering')
# # plotMethod(zOut, edgeList, 'AffinityPropagation')
# # plotMethod(zOut, edgeList, 'AgglomerativeClustering')
# # plotMethod(zOut, edgeList, 'AgglomerativeClusteringK')
# # plotMethod(zOut, edgeList, 'Birch')
# # plotMethod(zOut, edgeList, 'BirchN')
# # plotMethod(zOut, edgeList, 'MeanShift')
# # # plotMethod(zOut, edgeList, 'OPTICS')
# # plotMethod(zOut, edgeList, 'Leiden')

dataNameList = [
    '151507_cpm',
    '151508_cpm',
    '151509_cpm',
    '151510_cpm',
    '151669_cpm',
    '151670_cpm',
    '151671_cpm',
    '151672_cpm',
    '151673_cpm',
    '151674_cpm',
    '151675_cpm',
    '151676_cpm',
    '18-64_cpm',
    '2-5_cpm',
    '2-8_cpm',
    'T4857_cpm'
]

pe_typeList =['dummy','geom_lowf']
kList = ['10','50','100','200','500','1000','2000']
skStrList = ['8_Grid','16_GridEx',
            # '24_GridEx2','32_GridEx3',
            '40_GridEx4',
            # '48_GridEx5','56_GridEx6','64_GridEx7','72_GridEx8',
            '80_GridEx9',
            # '88_GridEx10','96_GridEx11','104_GridEx12','112_GridEx13',
            '120_GridEx14','160_GridEx19','200_GridEx24','240_GridEx29']

# For debug
# zOut,edgeList = readembedding(dataName='151671',k='',pe_type='',skStr='')
# clusteringMethod(zOut, edgeList, name='LouvainK', preK=5, resolution = 0.3)

# Single
# for dataName in dataNameList:
#     for pe_type in pe_typeList:
#         for k in kList:
#             for skStr in skStrList:                    
#                 #outputdirH-2-5_cpm_0.3/
#                 zOut,edgeList = readembedding(dataName,k,pe_type,skStr)
#                 clusteringMethod(zOut, edgeList, name='LouvainK', preK=5, resolution = 0.3)

for pe_type in pe_typeList:
    for k in kList:
        for skStr in skStrList:                    
            #outputdirH-2-5_cpm_0.3/
            zOut,edgeList = readembedding(args.dataName,k,pe_type,skStr)
            clusteringMethod(zOut, edgeList, name='LouvainK', preK=5, resolution = 0.3)







    
            

