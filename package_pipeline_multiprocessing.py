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
from multiprocessing import Pool, cpu_count
from case_study import case_study
warnings.filterwarnings("ignore")


def load_data(h5_path, spatial_path, scale_factor_path):
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

    return adata , spatial_all

def pseduo_images_scGNN(h5_path, spatial_path, scale_factor_path, output_folder,scgnnsp_zdim,scgnnsp_alpha,transform_opt):
    # --------------------------------------------------------------------------------------------------------#
    # -------------------------------load data--------------------------------------------------#
    # sample = h5_path.split('/')[-1].split('_')[0]
    sample = h5_path.split('/')[-2]
    adata,spatial_all = load_data(h5_path, spatial_path, scale_factor_path)
    # transform optional
    if transform_opt == 'log':
        sc.pp.log1p(adata)
    elif transform_opt == 'logcpm':
        sc.pp.normalize_total(adata,target_sum=1e4)
        sc.pp.log1p(adata)
    elif transform_opt == 'None':
        transform_opt = 'raw'
    else:
        print('transform optional is log or logcpm or None')

    #  panel gene
    # if panel_gene_path != None:  # case study
    #         gene_list = []
    #         with open(panel_gene_path, 'r') as f:
    #             for line in f:
    #                gene_list.append(line.strip())
    #         filter_panelgenes(adata,gene_list)

    print('load data finish')

    scgnnsp_knn_distanceList = ['euclidean']
    # scgnnsp_PEalphaList = ['0.1', '0.2', '0.3', '0.5', '1.0', '1.2', '1.5', '2.0']
    # scgnnsp_PEalphaList = [ '0.1',  '0.3', '0.5',  '1.0',  '1.5', '2.0']
    # scgnnsp_kList = ['6', '8']
    scgnnsp_kList = ['6']
    # scgnnsp_zdimList = ['3', '10', '16', '32', '64', '128', '256']
    scgnnsp_bypassAE_List = [True, False]
    # sample = 'scGNN'

    scgnnsp_bypassAE = scgnnsp_bypassAE_List[1]
    for scgnnsp_dist in scgnnsp_knn_distanceList:
        # for scgnnsp_zdim in scgnnsp_zdimList:
        #     for scgnnsp_alpha in scgnnsp_PEalphaList:
                for scgnnsp_k in scgnnsp_kList:
                    # --------------------------------------------------------------------------------------------------------#
                     # -------------------------------generate_embedding --------------------------------------------------#
                    # image_name = str(scgnnsp_k) + '_' + str(scgnnsp_dist) + '_' + str(
                    #     scgnnsp_alpha) + '_' + str(scgnnsp_zdim) + '_scGNN_LogCPM'
                    image_name =sample+'_scGNN_'+ transform_opt +'_PEalpha' +str(scgnnsp_alpha) +'_zdim'+str(scgnnsp_zdim)
                    # embedding = generate_embedding(adata,pca=pca, res=res,img_path = optical_img_path,method='spaGCN')
                    embedding = generate_embedding_sc(adata, sample=sample, scgnnsp_dist=scgnnsp_dist,
                                                      scgnnsp_alpha=scgnnsp_alpha, scgnnsp_k=scgnnsp_k,
                                                      scgnnsp_zdim=scgnnsp_zdim, scgnnsp_bypassAE=scgnnsp_bypassAE)
                    # embedding = embedding.detach().numpy()
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
    img_path = output_folder+ "/RGB_images/"
    inpaint_path = inpaint(img_path, sample, adata, spatial_all)
    print('generate pseudo images finish')
    return inpaint_path



# def pseudo_images(h5_path, spatial_path, scale_factor_path, output_folder,method, panel_gene_path, pca_opt,log_opt,normalization_opt):
def pseudo_images(h5_path, spatial_path, scale_factor_path, output_folder,method, panel_gene_path, pca_opt, transform_opt):
        # --------------------------------------------------------------------------------------------------------#
        # -------------------------------load data--------------------------------------------------#
    sample = h5_path.split('/')[-2]
    # print(sample)
    if method == 'spaGCN':
        adata,spatial_all = load_data(h5_path, spatial_path, scale_factor_path)


        if transform_opt == 'log':
            sc.pp.log1p(adata)
        elif transform_opt == 'logcpm':
            sc.pp.normalize_total(adata,target_sum=1e4)
            sc.pp.log1p(adata)
        elif transform_opt == 'None':
            transform_opt = 'raw'
        else:
            print('transform optional is log or logcpm or None')

        print('load data finish')

        pca_list = [32, 50, 64, 128, 256, 1024]
        res_list = np.arange(0.2, 0.7, 0.5)

        # panel gene
        # pca_opt = True
        if panel_gene_path != None:  # case study
            gene_list = []
            with open(panel_gene_path, 'r') as f:
                for line in f:
                   gene_list.append(line.strip())
            filter_panelgenes(adata,gene_list)
            pca_list = [3]
            res_list = [0.65]
            if pca_opt == False:
                pca_list = [0]
        else:
            # threshold
            threshold = 1.0
            adata.X[adata.X < threshold] = 0
            pca_opt = True


        # optical_img_path = os.path.join(data_path,"spatial/tissue_hires_image.png")
        optical_img_path = None
        for pca in pca_list:
            for res in res_list:
                # --------------------------------------------------------------------------------------------------------#
                # -------------------------------generate_embedding --------------------------------------------------#
                # image_name = str(pca) + '_'+str(res)+ '_spaGCN_LogCPM'
                image_name = sample+'_spaGCN_'+ transform_opt +'_pca' +str(pca) +'_res'+str(res)
                print(output_folder)
                print(image_name)
                embedding = generate_embedding_sp(adata,pca=pca, res=res,img_path = optical_img_path, pca_opt=pca_opt)
                embedding = embedding.detach().numpy()
                adata.obsm["embedding"] = embedding
                print('generate embedding finish')

                # --------------------------------------------------------------------------------------------------------#
                # --------------------------------transform_embedding_to_image-------------------------------------------------#
                high_img, low_img = transform_embedding_to_image(adata,image_name,output_folder,img_type='lowres',scale_factor_file=True)  # img_type:lowres,hires,both
                adata.uns["high_img"] = high_img
                adata.uns["low_img"] = low_img
                # adata.uns["img_shape"] = 600
                print('transform embedding to image finish')
        # # --------------------------------------------------------------------------------------------------------#
        # # --------------------------------inpaint image-------------------------------------------------#
        img_path = output_folder + "/RGB_images/"
        inpaint_path = inpaint(img_path, sample, adata, spatial_all)
        print('generate pseudo images finish')
        return inpaint_path

    elif method =='scGNN':
        scgnnsp_PEalphaList = [ '0.1', '0.2', '0.3', '0.5', '1.0', '1.2', '1.5', '2.0']
        scgnnsp_zdimList = ['3', '10',  '16',  '32', '64', '128', '256']

        core_num = cpu_count()
        # print(core_num)
        pool = Pool(core_num - 5)
        for scgnnsp_zdim in scgnnsp_zdimList:
                # for scgnnsp_dist in scgnnsp_knn_distanceList:
                    for scgnnsp_alpha in scgnnsp_PEalphaList:
                        # for scgnnsp_k in scgnnsp_kList:
                        pool.apply_async(pseduo_images_scGNN, (h5_path, spatial_path, scale_factor_path, output_folder,
                                                               scgnnsp_zdim,scgnnsp_alpha,transform_opt,))
                        # pseduo_images_scGNN(h5_path, spatial_path, scale_factor_path, output_folder,
                        #                                        scgnnsp_zdim,scgnnsp_alpha,transform_opt)
        pool.close()
        pool.join()


def segmentation_test(h5_path, spatial_path, scale_factor_path, output_path, method,panel_gene_path,pca_opt,transform_opt,checkpoint, device):
    pseudo_images(h5_path, spatial_path, scale_factor_path, output_path, method,panel_gene_path,pca_opt,transform_opt)   # output_folder+ "/pseudo_images/"
    img_path = output_path + "/RGB_images/"
    label_path = None
    adata = None
    top1_csv_name= segmentation(adata,img_path,label_path,method,checkpoint, device)
    return top1_csv_name


def segmentation_category_map(h5_path, spatial_path, scale_factor_path, optical_path, output_path, method, panel_gene_path, pca_opt, transform_opt, checkpoint, device):
    optical_img = cv2.imread(optical_path)
    category_map = segmentation_test(h5_path, spatial_path, scale_factor_path, output_path, method, panel_gene_path, pca_opt, transform_opt, checkpoint, device)
#     category_map = np.loadtxt(top1_csv_name,dtype=np.int32, delimiter=",")  
    seg_category_map(optical_img, category_map, output_path)


def segmentation_evaluation(h5_path, spatial_path, scale_factor_path, output_path, method,label_path, panel_gene_path,pca_opt,transform_opt,checkpoint, device):
    pseudo_images(h5_path, spatial_path, scale_factor_path, output_path, method, panel_gene_path,pca_opt,transform_opt)
    img_path =output_path + "/RGB_images/"
    adata,spatial_all = load_data(h5_path, spatial_path, scale_factor_path)
    adata.uns["img_shape"] = 600
    top1_csv_name= segmentation(adata,img_path,label_path,method,checkpoint, device)


def case_study_test(h5_path, spatial_path, scale_factor_path, output_path, method,  panel_gene_path , pca_opt, transform_opt,r_tuple,g_tuple,b_tuple):
    img_path = pseudo_images(h5_path, spatial_path, scale_factor_path, output_path, method,  panel_gene_path , pca_opt, transform_opt)   # output_folder+ "/pseudo_images/"
    case_study(img_path,r_tuple,g_tuple,b_tuple)
