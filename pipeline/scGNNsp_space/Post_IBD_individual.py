import numpy as np
import pandas as pd
import os
from benchmark_util import *
from sklearn.metrics.cluster import adjusted_rand_score
import argparse

current_path = os.path.abspath('.')


parser = argparse.ArgumentParser(description='arg')
parser.add_argument('--dir', type=str, default='/outputdir/',
                    help='outputdir')
parser.add_argument('--inputfile', type=str, default='151673_all_results',
                    help='151673_all_results.txt')
parser.add_argument('--datasetname', type=str, default='151673',
                    help='dataset name')
args = parser.parse_args()

fileFolder = current_path+'/'+args.datasetname+'/'


labelname = fileFolder+'label.csv'
# labelname = current_path+'/outputdir_ccG/151673_noregu_0.9_1.0_0.0_results_6.txt'
# resultname = current_path+'/outputdir/151673_dummy_add_0.0_results_5.txt'
# resultname = current_path+'/outputdir/151673_all_results.txt'
# resultname = current_path+'/outputdir/151673_geom_lowf_add_2.0_results_7.txt'
# resultname = current_path+'/outputdirscgnn/151673_noregu_0.9_1.0_0.0_results_9.txt'

resultname = current_path+args.dir+args.inputfile+'.txt'

df = pd.read_csv(labelname)
listBench = df['layer'].to_numpy().tolist()
# listBench = df['Celltype'].to_numpy().tolist()

df = pd.read_csv(resultname)
listResult = df['Celltype'].to_numpy().tolist()

ari, ami, nmi, cs, fms, vms, hs = measureClusteringTrueLabel(listBench, listResult)
resultstr = args.inputfile+' '+str(ari)+' ' + \
            str(ami)+' '+str(nmi)+' '+str(cs)+' ' + \
            str(fms)+' '+str(vms)+' '+str(hs)
print(resultstr)
