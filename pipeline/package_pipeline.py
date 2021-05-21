import os,csv,re,json
import pandas as pd
import numpy as np
import scanpy as sc
import math
# import SpaGCN as spg
import cv2
from skimage import io, color
# from SpaGCN.util import prefilter_specialgenes, prefilter_genes
from pipeline_transform_spaGCN_embedding_to_image import transform_embedding_to_image
from generate_embedding import generate_embedding
from util import filter_panelgenes
import random, torch
from test import segmentation
from inpaint_images import inpaint
import warnings
import argparse
import glob
from find_category import seg_category_map
warnings.filterwarnings("ignore")


def pseudo_images(h5_path, spatial_path, scale_factor_path, output_folder, panel_gene_path= None):
    # --------------------------------------------------------------------------------------------------------#
    # -------------------------------load data--------------------------------------------------#
    # Read in gene expression and spatial location
    adata = sc.read_10x_h5(h5_path)
    spatial_all=pd.read_csv(spatial_path,sep=",",header=None,na_filter=False,index_col=0)
    spatial = spatial_all[spatial_all[1] == 1]
    spatial = spatial.sort_values(by=0)
    assert all(adata.obs.index == spatial.index)
    adata.obs["in_tissue"]=spatial[1]
    adata.obs["array_row"]=spatial[2]
    adata.obs["array_col"]=spatial[3]
    adata.obs["pxl_col_in_fullres"]=spatial[4]
    adata.obs["pxl_row_in_fullres"]=spatial[5]
    adata.obs.index.name = 'barcode'
    adata.var_names_make_unique()


    # Read scale_factor_file
    with open(scale_factor_path) as fp_scaler:
        scaler = json.load(fp_scaler)
    adata.uns["spot_diameter_fullres"] = scaler["spot_diameter_fullres"]
    adata.uns["tissue_hires_scalef"] = scaler["tissue_hires_scalef"]
    adata.uns["fiducial_diameter_fullres"] = scaler["fiducial_diameter_fullres"]
    adata.uns["tissue_lowres_scalef"] = scaler["tissue_lowres_scalef"]


    # logcpm
    sc.pp.normalize_total(adata,target_sum=1e4)
    sc.pp.log1p(adata)

    # threshold
    threshold = 1.0
    adata.X[adata.X < threshold]= 0
    # pca_list = [32]
    pca_list = [32, 50, 64, 128, 256, 1024]
    res_list = np.arange(0.2, 0.7, 0.05)

    # panel gene
    pca_opt = True
    if panel_gene_path != None:
        gene_list = []
        with open(gene_path, 'r') as f:
            for line in f:
               gene_list.append(line.strip())
        filter_panelgenes(adata,gene_list)
        pca_list = [50]
        res_list = [0.65]
        if len(gene_list) < 50:
            pca_opt = False
    print('load data finish')  



    # optical_img_path = os.path.join(data_path,"spatial/tissue_hires_image.png")
    optical_img_path = None
    for pca in pca_list:
        for res in res_list:
            # --------------------------------------------------------------------------------------------------------#
            # -------------------------------generate_embedding --------------------------------------------------#
            image_name = str(pca) + '_'+str(res)+ '_spa_LogCPM'
            embedding = generate_embedding(adata,pca=pca, res=res,img_path = optical_img_path, pca_opt=pca_opt, method='spaGCN')
            embedding = embedding.detach().numpy()
            adata.obsm["embedding"] = embedding
            print('generate embedding finish')

            # --------------------------------------------------------------------------------------------------------#
            # --------------------------------transform_embedding_to_image-------------------------------------------------#
            high_img, low_img = transform_embedding_to_image(adata,image_name,output_folder,img_type='lowres',scale_factor_file=True)  # img_type:lowres,hires,both
            adata.uns["high_img"] = high_img
            adata.uns["low_img"] = low_img
            print('transform embedding to image finish')

    # --------------------------------------------------------------------------------------------------------#
    # --------------------------------inpaint image-------------------------------------------------#
    img_path = output_folder+ "/pseudo_images/"
    inpaint_path = inpaint(img_path, adata, spatial_all)
    print('generate pseudo images finish')
    return inpaint_path

def segmentation_test(h5_path, spatial_path, scale_factor_path, output_path):
    img_path = pseudo_images(h5_path, spatial_path, scale_factor_path, output_path)   # output_folder+ "/pseudo_images/"
    top1_csv_name= segmentation(img_path)
    return top1_csv_name


def segmentation_category_map(h5_path, spatial_path, scale_factor_path, optical_path, output_path):
    optical_img = cv2.imread(optical_path)
    top1_csv_name = segmentation_test(h5_path, spatial_path, scale_factor_path, output_path)
    category_map = np.loadtxt(top1_csv_name,dtype=np.int32, delimiter=",")  
    seg_category_map(optical_img, category_map, output_path)