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
from generate_embedding import generate_embedding_sp,generate_embedding_sc
from util import filter_panelgenes
import random, torch
from test import segmentation
from inpaint_images import inpaint
import warnings
import argparse
import glob
from find_category import seg_category_map
from case_study import case_study

warnings.filterwarnings("ignore")


def pseudo_images(h5_path, spatial_path, scale_factor_path, output_folder,method='scGNN', panel_gene_path=None):
    if method == 'spaGCN':
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
        adata.X[adata.X < threshold] = 0

        # pca_list = [32]
        print('load data finish')


        pca_list = [32, 50, 64, 128, 256, 1024]
        res_list = np.arange(0.2, 0.7, 0.05)

        # panel gene
        pca_opt = True
        if panel_gene_path != None:
            gene_list = []
            with open(panel_gene_path, 'r') as f:
                for line in f:
                   gene_list.append(line.strip())
            filter_panelgenes(adata,gene_list)
            pca_list = [50]
            res_list = [0.65]
            if len(gene_list) < 50:
                pca_opt = False


        # optical_img_path = os.path.join(data_path,"spatial/tissue_hires_image.png")
        optical_img_path = None
        for pca in pca_list:
            for res in res_list:
                # --------------------------------------------------------------------------------------------------------#
                # -------------------------------generate_embedding --------------------------------------------------#
                image_name = str(pca) + '_'+str(res)+ '_spa_LogCPM'
                embedding = generate_embedding_sp(adata,pca=pca, res=res,img_path = optical_img_path, pca_opt=pca_opt)
                embedding = embedding.detach().numpy()
                adata.obsm["embedding"] = embedding
                print('generate embedding finish')

                # --------------------------------------------------------------------------------------------------------#
                # --------------------------------transform_embedding_to_image-------------------------------------------------#
                high_img, low_img = transform_embedding_to_image(adata,image_name,output_folder,img_type='lowres',scale_factor_file=True)  # img_type:lowres,hires,both
                adata.uns["high_img"] = high_img
                adata.uns["low_img"] = low_img
                print('transform embedding to image finish')

    elif method =='scGNN':


        # --------------------------------------------------------------------------------------------------------#
        # -------------------------------load data--------------------------------------------------#
        # Read in gene expression and spatial location
        adata = sc.read_10x_h5(h5_path)
        spatial_all = pd.read_csv(spatial_path, sep=",", header=None, na_filter=False, index_col=0)
        spatial = spatial_all[spatial_all[1] == 1]
        spatial = spatial.sort_values(by=0)
        assert all(adata.obs.index == spatial.index)
        adata.obs["in_tissue"] = spatial[1]
        adata.obs["array_row"] = spatial[2]
        adata.obs["array_col"] = spatial[3]
        adata.obs["pxl_col_in_fullres"] = spatial[4]
        adata.obs["pxl_row_in_fullres"] = spatial[5]
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
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # threshold
        # threshold = 1.0
        # adata.X[adata.X < threshold] = 0

        # pca_list = [32]
        print('load data finish')

        # scgnnsp_knn_distanceList = ['euclidean', 'cityblock']
        scgnnsp_knn_distanceList = ['euclidean']
        scgnnsp_PEalphaList = [ '0.1', '0.2', '0.3', '0.5', '1.0', '1.2', '1.5', '2.0']
        # scgnnsp_PEalphaList = [ '0.1',  '0.3', '0.5',  '1.0',  '1.5', '2.0']
        # scgnnsp_kList = ['6', '8']
        scgnnsp_kList = ['6']
        scgnnsp_zdimList = ['3', '10',  '16',  '32', '64', '128', '256']
        scgnnsp_bypassAE_List = [True, False]
        sample = 'sc'

        # panel gene
        scgnnsp_bypassAE = scgnnsp_bypassAE_List[1]
        if panel_gene_path != None:
            gene_list = []
            with open(panel_gene_path, 'r') as f:
                for line in f:
                    gene_list.append(line.strip())
            filter_panelgenes(adata, gene_list)
            scgnnsp_knn_distanceList = ['euclidean']
            scgnnsp_PEalphaList = ['0.5']
            scgnnsp_kList = ['6']
            scgnnsp_zdimList = []
            scgnnsp_zdimList.append(str(len(gene_list)))
            scgnnsp_bypassAE = scgnnsp_bypassAE_List[0]


        for scgnnsp_zdim in scgnnsp_zdimList:
            for scgnnsp_dist in scgnnsp_knn_distanceList:
                for scgnnsp_alpha in scgnnsp_PEalphaList:
                    for scgnnsp_k in scgnnsp_kList:
                        # --------------------------------------------------------------------------------------------------------#
                        # -------------------------------generate_embedding --------------------------------------------------#
                        image_name = sample + '_'+str(scgnnsp_k)+ '_'+str(scgnnsp_dist) + '_'+str(scgnnsp_alpha)+'_'+str(scgnnsp_zdim)+ '_sc_LogCPM'
                        #embedding = generate_embedding(adata,pca=pca, res=res,img_path = optical_img_path,method='spaGCN')
                        embedding = generate_embedding_sc(adata, sample=sample, scgnnsp_dist=scgnnsp_dist, scgnnsp_alpha=scgnnsp_alpha, scgnnsp_k=scgnnsp_k,scgnnsp_zdim=scgnnsp_zdim, scgnnsp_bypassAE=scgnnsp_bypassAE)
                        #embedding = embedding.detach().numpy()
                        adata.obsm["embedding"] = embedding
                        print('generate embedding finish')
                        os.getcwd()

                        # --------------------------------------------------------------------------------------------------------#
                        #         # --------------------------------transform_embedding_to_image-------------------------------------------------#
                        high_img, low_img = transform_embedding_to_image(adata, image_name, output_folder,
                                                                         img_type='lowres',
                                                                         scale_factor_file=True)  # img_type:lowres,hires,both
                        adata.uns["high_img"] = high_img
                        adata.uns["low_img"] = low_img
                        print('transform embedding to image finish')

    # --------------------------------------------------------------------------------------------------------#
    # --------------------------------inpaint image-------------------------------------------------#
    img_path = output_folder+ "/pseudo_images/"
    inpaint_path = inpaint(img_path, adata, spatial_all)
    print('generate pseudo images finish')
    return inpaint_path

def segmentation_test(h5_path, spatial_path, scale_factor_path, output_path, method,panel_gene_path):
    img_path = pseudo_images(h5_path, spatial_path, scale_factor_path, output_path, method,panel_gene_path)   # output_folder+ "/pseudo_images/"
    top1_csv_name= segmentation(img_path,method)
    return top1_csv_name


def segmentation_category_map(h5_path, spatial_path, scale_factor_path, optical_path, output_path, method):
    optical_img = cv2.imread(optical_path)
    top1_csv_name = segmentation_test(h5_path, spatial_path, scale_factor_path, output_path, method)
    category_map = np.loadtxt(top1_csv_name,dtype=np.int32, delimiter=",")  
    seg_category_map(optical_img, category_map, output_path)


def case_study_test(h5_path, spatial_path, scale_factor_path, output_path, method,  panel_gene_path , pca_opt, transform_opt,r_tuple,g_tuple,b_tuple):
    img_path = pseudo_images(h5_path, spatial_path, scale_factor_path, output_path, method,  panel_gene_path , pca_opt, transform_opt)   # output_folder+ "/pseudo_images/"
    case_study(img_path,r_tuple,g_tuple,b_tuple)
