# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 14:26:27 2021

@author: bioinf
"""
import scanpy as sc
import os, sys
import pandas as pd
import numpy as np
import warnings
import SpaGCN as spg
import random, torch
warnings.filterwarnings("ignore")
from scipy.sparse.csr import csr_matrix
from anndata import AnnData
import shutil
from pathlib import Path
import time, resource
import csv
# for an given expression, find the corressponding position index in position dir
def pos_find(exp_name, pos_name_list):
    cur = str.split(spe_exp_data[j], sep='_')[0]
    total_index = []
    for t in range(len(pos_name_list)):
        temp = str.split(pos_name_list[t], sep='_')[0]
        total_index.append(temp)
    return total_index.index(cur)

# obtain files in given path containing pattern
def finder(path, pattern=""):
    matches = []
    for x in os.listdir(path):
        if pattern in x:
            matches.append(x)
    return matches

# here****************************************************
def time_mem_save(job_name, time_begin, csv_name):
    # explain
    # job_name: current job name
    # time_begin: the system time at the beginning
    # csv_name: the file name including path, which stores running information

    # obtain run time
    time_end = time.time()
    time_run = time_end - time_begin

    # obtain mem consume
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    # save information
    csvFile = open(csv_name, 'a')
    writer = csv.writer(csvFile)
    information = "success at:" + str(job_name) + ";" + "mem:" + str(mem) + ";" + "time:" + str(time_run)
    writer.writerow([information])

time_begin = time.time()
csv_name = '//scratch/10x/Jixin_script/results/mem_time.csv'

all_exp_dir = "//scratch/10x/Human_brain_JH_12/normalized_data/" # format npz
all_pos_dir = "//scratch/10x/Human_brain_JH_12/sparse_meta_out/"
save_dir = "//scratch/10x/Jixin_script/results/SpaGCN/"

all_exp_normName = [sys.argv[1]]
all_pos_file = finder(all_pos_dir, pattern="metaData.csv")

for i in range(len(all_exp_normName)):
    spe_exp_dir = all_exp_dir + all_exp_normName[i] + "/spa_out"
    pattern = sys.argv[2] + "_humanBrain_" + sys.argv[1] +  "_spa.npz"
    spe_exp_data = finder(spe_exp_dir, pattern=pattern)
    pos_data = all_pos_file
    
    for j in range(len(spe_exp_data)):
    # read expression or velocity data
        try:
            # here ***************************************************************
            save_adj_dir = "//scratch/10x/Jixin_script/temp_files/ADJ/" # temp dir
            save_HD_dir = "//scratch/10x/Jixin_script/temp_files/HD/" # temp dir
            
            save_adj_dir = save_adj_dir + all_exp_normName[i] + '/' + spe_exp_data[j].split(".n")[0] + "/defalut" + "/SpaGCN/"
            save_adj1_dir = Path(save_adj_dir)
            save_adj1_dir.mkdir(parents=True, exist_ok=True)
            
            save_HD_dir = save_HD_dir + all_exp_normName[i] + '/' + spe_exp_data[j].split(".n")[0] + "/default" + "/SpaGCN/"
            save_HD1_dir = Path(save_HD_dir)
            save_HD1_dir.mkdir(parents=True, exist_ok=True)
            # create adata object
            exp_dir = spe_exp_dir + "/" + spe_exp_data[j] 
            temp_data = np.load(exp_dir)
            pIndex = pos_find(spe_exp_data[j], pos_data)
            spe_pos_dir = all_pos_dir + pos_data[pIndex]
            spatial = pd.read_csv(spe_pos_dir)
            
            e = csr_matrix((temp_data['data'], temp_data['indices'], temp_data['indptr']), dtype=float)
            f = AnnData(e)
            adata = f.transpose()
            spatial.index._name = 'barcode'
            adata.obs = pd.DataFrame(index=spatial["barcode"])
            adata.obs["x1"] = list(spatial["in_tissue"])
            adata.obs["x2"] = list(spatial["array_row"])
            adata.obs["x3"] = list(spatial["array_col"])
            adata.obs["x4"] = list(spatial["pxl_col_in_fullres"])
            adata.obs["x5"] = list(spatial["pxl_row_in_fullres"])
            adata = adata[adata.obs["x1"] == 1]

            adata.var_names = [i.upper() for i in list(adata.var_names)]
            adata.var["genename"] = adata.var.index.astype("str")
            adata.write_h5ad(save_HD_dir+"temp.h5ad")
            adata=sc.read(save_HD_dir+"temp.h5ad")
            
            # Calculate adjacent matrix
            b=49
            a=1
            print("here")
            
            # Spot coordinates
            x2=adata.obs["x2"].tolist()
            x3=adata.obs["x3"].tolist()
            
            # Pixel coordinates
            x4=adata.obs["x4"].tolist()
            x5=adata.obs["x5"].tolist()
            adj = spg.calculate_adj_matrix(x=x2, y=x3, x_pixel=x4, y_pixel=x5, image=None, beta=b, alpha=a, histology=False)
            
            np.savetxt(save_adj_dir + "temp.csv" , adj, delimiter=',')
            
            # Set seed
            random.seed(200)
            torch.manual_seed(200)
            np.random.seed(200)
            
            adata=sc.read(save_HD_dir+"temp.h5ad")
            adj = np.loadtxt(save_adj_dir + "temp.csv", delimiter=',')
            adata.var_names_make_unique()
            
            if not (sys.argv[1] == "velocity"):
                spg.prefilter_genes(adata)  # avoiding all genes are zeros
                spg.prefilter_specialgenes(adata)
            p = 0.5
            # l = spg.find_l(p=p,adj=adj,start=1, end=1.6,sep=0.01, tol=0.01)
            l = 1.43
            clf=spg.SpaGCN()
            clf.set_l(l)        
            clf.train(adata,adj,init_spa=True,init="louvain", num_pcs=128, louvain_seed=0)
            y_pred, prob=clf.predict()
            
            results = pd.DataFrame({'barcode':adata.obs.index, "label":y_pred})
            # here *************************************************************************************
            results.to_csv(save_dir + str.split(spe_exp_data[j], sep='_')[0] + "_SpaGCN_" + all_exp_normName[i] + "_default.csv", index=False)
            
            # obtain running time as well as memory consumption
            job_name = str.split(spe_exp_data[j], sep='_')[0] + "_SpaGCN_" + all_exp_normName[i] + "_default.csv"
            time_mem_save(job_name, time_begin, csv_name)
            print('mem_time correctly')
            shutil.rmtree(save_HD_dir)
            print("remove temp files correctly")
            shutil.rmtree(save_adj_dir)
        except Exception as e:
            print(e)
