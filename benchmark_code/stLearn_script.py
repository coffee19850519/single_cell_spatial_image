# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 21:28:15 2021

@author: bioinf
"""

import stlearn as st
from pathlib import Path
import pandas as pd
import os, sys
import numpy as np
import shutil
import time
import resource
import csv

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
csv_name = '//scratch/10x/Jixin_script/results/mem_time_2.csv'

all_data_dir = "//scratch/10x/original/" # a dir contains all data, including expression, image, positon
# all_data_dir_list = os.listdir(all_data_dir)
all_data_dir_list = [sys.argv[1]]
save_dir = "//scratch/10x/Jixin_script/results/stLearn/individual" # the results will be stored in this dir
image_dir = "//scratch/10x/Jixin_script/temp_files/IMAGE/" # temp dir, usuless
out_dir = "//scratch/10x/Jixin_script/temp_files/OUT/" # temp dir, useless

for j in range(len(all_data_dir_list)):
    try:
        # specify PATH to data
        spe_data_dir = all_data_dir + all_data_dir_list[j]
        BASE_PATH = Path(spe_data_dir)
      
        # spot tile is the intermediate result of image pre-processing
        spe_image = image_dir + all_data_dir_list[j] + "/default/"  + "/stLearn/"
        TILE_PATH = Path(spe_image)
        TILE_PATH.mkdir(parents=True, exist_ok=True)
        
        # pre-processing for gene count table
        spe_out = out_dir + all_data_dir_list[j] + "/defalut/" + "/stLearn/"
        OUT_PATH = Path(spe_out)
        OUT_PATH.mkdir(parents=True, exist_ok=True)
        
        data = st.Read10X(BASE_PATH)
    
        # pre-processing for gene count table
        st.pp.filter_genes(data, min_cells=1)
        st.pp.normalize_total(data)
        st.pp.log1p(data)
        
        # pre-processing for spot image
        st.pp.tiling(data, TILE_PATH)
        
        # this step uses deep learning model to extract high-level features from tile images
        # may need few minutes to be completed
        st.pp.extract_feature(data, n_components=128)
        
        # run PCA for gene expression data
        st.em.run_pca(data, n_comps=128)
        
        data_SME = data.copy()
        # apply stSME to normalise log transformed data
        st.spatial.SME.SME_normalize(data_SME, use_data="raw")
        data_SME.X = data_SME.obsm['raw_SME_normalized']
        st.pp.scale(data_SME)
        st.em.run_pca(data_SME, n_comps=128)
        
        # # K-means clustering on stSME normalised PCA
        # st.tl.clustering.kmeans(data_SME,n_clusters=19, use_data="X_pca", key_added="X_pca_kmeans")
        # st.pl.cluster_plot(data_SME, use_label="X_pca_kmeans")
        
        # louvain clustering on stSME normalised data
        st.pp.neighbors(data_SME, use_rep='X_pca')
        st.tl.clustering.louvain(data_SME)
        st.pl.cluster_plot(data_SME, use_label="louvain")
        
        # save cluster results
        save_dir_name = save_dir + all_data_dir_list[j] +"_stLearn_None_" + "default" + ".csv"
        result = pd.DataFrame(data_SME.obs['louvain'])
        label = list(result.louvain)
        result = {"barcode": list(result.index), "label" : label}
        result = pd.DataFrame(result)
        result.to_csv(save_dir_name, index=False)

        # obtain running time as well as memory consumption
        job_name = all_data_dir_list[j] +"_stLearn_None_" + "default" + ".csv"
        time_mem_save(job_name, time_begin, csv_name)
    except Exception as e:
        print(e)
        print("default")
        print(spe_data_dir)
    # delete temp files
    shutil.rmtree(spe_image)
    shutil.rmtree(spe_out)

        

    
    
    
