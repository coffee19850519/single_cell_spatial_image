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
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='from panel gene generate pseudo images')
    parser.add_argument('sample_list', type=str, nargs='+', help='input sample list')
    parser.add_argument('-p','--panelgene_list',type=str,nargs='+',help='input panel gene list')
    parser.add_argument('-r', '--resolution', type=float,nargs='*', default=0.65,
                        help='default:0.65 ,If you want resolution range, set -rl')
    parser.add_argument('-rl', '--resolution_list', type=float,nargs='*',
                        help='input range first,last and step.Like:0.2 0.7 0.05')
    parser.add_argument('-ip', '--input_path', type=str, nargs='*', default='original',
                        help='original 10x file folder')
    parser.add_argument('-op', '--output_path', type=str, nargs='*',default='generate_pseudo_images_panel_test',
                        help='generate pseudo images folder')
    args = parser.parse_args()
    return args


args = parse_args()
res_list = []
if args.resolution_list == None:
    res_list.append(args.resolution)
else:
    res_list = np.arange(args.resolution_list[0], args.resolution_list[1], args.resolution_list[2])


#
pca_list = [32]
sample_list = args.sample_list
gene_list = args.panelgene_list
for sample in sample_list:
    data_path = args.input_path+'/'+sample+"/"
    # data_path = "othertissue_original/" + sample + "/"
    h5_path = glob.glob(data_path+"*.h5")[0]
    scale_factor_path = "spatial/scalefactors_json.json"
    spatial_path = "spatial/tissue_positions_list.csv"
    pseudo_image_folder = args.output_path+'/'
    if not os.path.exists(pseudo_image_folder):
        os.makedirs(pseudo_image_folder)

    # --------------------------------------------------------------------------------------------------------#
    # -------------------------------load data--------------------------------------------------#
    # Read in gene expression and spatial location
    adata = sc.read_10x_h5(h5_path)
    spatial_all=pd.read_csv(os.path.join(data_path,spatial_path),sep=",",header=None,na_filter=False,index_col=0)
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
    with open(os.path.join(data_path,scale_factor_path)) as fp_scaler:
        scaler = json.load(fp_scaler)
    adata.uns["spot_diameter_fullres"] = scaler["spot_diameter_fullres"]
    adata.uns["tissue_hires_scalef"] = scaler["tissue_hires_scalef"]
    adata.uns["fiducial_diameter_fullres"] = scaler["fiducial_diameter_fullres"]
    adata.uns["tissue_lowres_scalef"] = scaler["tissue_lowres_scalef"]


    # logcpm
    sc.pp.normalize_total(adata,target_sum=1e4)
    sc.pp.log1p(adata)

    # panel gene
    filter_panelgenes(adata,gene_list)
    print('load data finish')


    # optical_img_path = os.path.join(data_path,"spatial/tissue_hires_image.png")
    optical_img_path = None
    sample_name = data_path.split('/')[1]
    for pca in pca_list:
        for res in res_list:

            # --------------------------------------------------------------------------------------------------------#
            # -------------------------------generate_embedding --------------------------------------------------#
            image_name = sample_name + '_'+str(res)+ '_spa_LogCPM'
            embedding = generate_embedding(adata,pca=pca, res=res,img_path = optical_img_path,method='spaGCN')
            embedding = embedding.detach().numpy()
            adata.obsm["embedding"] = embedding
            print('generate embedding finish')

            # --------------------------------------------------------------------------------------------------------#
            # --------------------------------transform_embedding_to_image-------------------------------------------------#

            high_img, low_img = transform_embedding_to_image(adata,image_name,pseudo_image_folder,img_type='lowres',scale_factor_file=True)  # img_type:lowres,hires,both
            adata.uns["high_img"] = high_img
            adata.uns["low_img"] = low_img
    print('transform embedding to image finish')

    # --------------------------------------------------------------------------------------------------------#
    # --------------------------------inpaint image-------------------------------------------------#
    img_path = pseudo_image_folder+sample_name+"/pseudo_image/"
    inpaint_path = inpaint(img_path,adata,spatial_all)
    print('inpaint image finish')


