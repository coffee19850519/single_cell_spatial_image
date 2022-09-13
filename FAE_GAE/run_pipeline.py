import pandas as pd
import numpy as np
import torch
import warnings
import time
import argparse
import os,csv,re,json
import scanpy as sc
from data_processing import generate_coords_sc
# from produce_PAE_embedding import feature_AE
from gae_embedding import GAEembedding
from GAEdata_prepare import coords_to_adj, load_data
from pipeline_transform_spaGCN_embedding_to_image import transform_embedding_to_image
from inpaint_images import inpaint
# from multiprocessing import Pool, cpu_count


warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='the mode of scGNN picture')
    parser.add_argument('-sample', type=str, nargs='+', default=['151507'], help='which sample to generate: 151507-101510,151669-151676,2-5,2-8,18-64,T4857.')
    parser.add_argument('-enc_type', type=str, nargs='+', default=['dummy'], help='the type of feature encoder:dummy, geom_ft, geom_full, geom_lowf, geom_nohighf or linear_lowf')
    args = parser.parse_args()
    return args

def pseduo_images_scGNN(h5_path, spatial_path, scale_factor_path, output_folder, scgnnsp_zdim, scgnnsp_alpha, transform_opt, zDiscret_mode, GAEmodel,enc_type):
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
    print('load data finish')

    scgnnsp_knn_distanceList = ['euclidean']
    scgnnsp_kList = ['6']
    scgnnsp_bypassAE_List = [True, False]
    scgnnsp_bypassAE = scgnnsp_bypassAE_List[1]
    for scgnnsp_dist in scgnnsp_knn_distanceList:
        for scgnnsp_k in scgnnsp_kList:

            image_name =sample+'_scGNN_'+ transform_opt +'_PEalpha' +str(scgnnsp_alpha) +'_zdim'+str(scgnnsp_zdim) +'_'+str(zDiscret_mode)+'_'+str(GAEmodel)
            print('------------------image_name-----------------------', image_name)

            #data preprocessing
            coords, ues_expression = generate_coords_sc(adata, sample=sample, scgnnsp_dist=scgnnsp_dist,
                                              scgnnsp_alpha=scgnnsp_alpha, scgnnsp_k=scgnnsp_k,
                                              scgnnsp_zdim=scgnnsp_zdim, scgnnsp_bypassAE=scgnnsp_bypassAE)

            z = ues_expression.to_numpy()
            z = z.T
            zOut = z

            #----------Graph+Feature Autoencoder----------#
            if zDiscret_mode =='mean':
                zDiscret = zOut > np.mean(zOut, axis=0)
                zDiscret = 1.0*zDiscret
            elif zDiscret_mode == 'median':
                zDiscret = zOut > np.median(zOut, axis=0)
                zDiscret = 1.0*zDiscret
            else:
                zDiscret = zOut

            adj, edgeList = coords_to_adj(zDiscret,coords)

            embedding = GAEembedding(zDiscret, adj, GAEmodel, scgnnsp_alpha, scgnnsp_zdim, enc_type, coords)

            ###
            embedding = embedding.detach().numpy()

            adata.obsm["embedding"] = embedding
            print('generate embedding finish')
            high_img, low_img = transform_embedding_to_image(adata, image_name, output_folder,
                                                             img_type='lowres',
                                                             scale_factor_file=True)  # img_type:lowres,hires,both
            adata.uns["high_img"] = high_img
            adata.uns["low_img"] = low_img
            print('transform embedding to image finish')
    img_path = output_folder + '/RGB_images/'
    print('-------------img_path---------------', img_path)
    inpaint_path = inpaint(img_path, sample, adata, spatial_all)
    print('generate pseudo images finish')
    return inpaint_path



def pseudo_images(h5_path, spatial_path, scale_factor_path, output_folder, method, transform_opt,zDiscret_mode,GAEmodel,enc_type):
    sample = h5_path.split('/')[-2]
    print(sample)

    method == 'scGNN'
    scgnnsp_PEalphaList = ['0.1']       #['0.1','0.2', '0.3', '0.5', '1.0', '1.2', '1.5', '2.0']
    scgnnsp_zdimList = ['10']           #['10', '16', '32', '64', '128', '256']

    #-------------------Single process-----------------#
    for scgnnsp_zdim in scgnnsp_zdimList:
        # for scgnnsp_dist in scgnnsp_knn_distanceList:
        print('zdim------------------->',scgnnsp_zdim)
        for scgnnsp_alpha in scgnnsp_PEalphaList:
            # for scgnnsp_k in scgnnsp_kList:
            pseduo_images_scGNN(h5_path, spatial_path, scale_factor_path, output_folder,scgnnsp_zdim, scgnnsp_alpha, transform_opt,zDiscret_mode, GAEmodel,enc_type)
    #--------------------Multi process-----------------#
    # core_num = cpu_count()
    # print(core_num)
    # pool = Pool(core_num - 5)
    # for scgnnsp_zdim in scgnnsp_zdimList:
    #     for scgnnsp_alpha in scgnnsp_PEalphaList:
    #         pool.apply_async(pseduo_images_scGNN, (h5_path, spatial_path, scale_factor_path, output_folder,scgnnsp_zdim, scgnnsp_alpha, transform_opt,label_path,epochs,enc_type,zDiscret_mode, GAEmodel))
    # pool.close()
    # pool.join()

if __name__ == "__main__":
    args = parse_args()
    sample = args.sample[0]
    enc_type = args.enc_type[0]
    h5_path = 'E:/RESEPT_1/16set_data/'+sample+'/filtered_feature_bc_matrix.h5'
    spatial_path = 'E:/RESEPT_1/16set_data/'+sample+'/spatial/tissue_positions_list.csv'
    scale_factor_path = 'E:/RESEPT_1/16set_data/'+sample+'/spatial/scalefactors_json.json'
    zDiscret_mode_list = ['mean']#, 'median',None]
    GAEmodel_list = ['vgae']#, 'gae']

    out_put_path = 'result'     ####
    if not os.path.exists(out_put_path):
        os.makedirs(out_put_path)

    for zDiscret_mode in zDiscret_mode_list:
        for GAEmodel in GAEmodel_list:
            pseudo_images(h5_path, spatial_path, scale_factor_path, out_put_path, 'scGNN', 'logcpm', zDiscret_mode, GAEmodel,enc_type)
