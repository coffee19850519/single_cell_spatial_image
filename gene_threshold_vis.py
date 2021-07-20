import os,csv,re,json
import pandas as pd
import numpy as np
import scanpy as sc
import math
# import SpaGCN as spg
import cv2
from skimage import io, color
# from SpaGCN.util import prefilter_specialgenes, prefilter_genes
from util import filter_panelgenes
import random, torch
from test import segmentation
from inpaint_images import inpaint
import warnings
import argparse
import glob
from find_category import seg_category_map
from math import sqrt
warnings.filterwarnings("ignore")


def expression_threshold_visulization(h5_path, spatial_path, scale_factor_path, output_folder, panel_gene_path):
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

    gene_list = []
    with open(panel_gene_path, 'r') as f:
        for line in f:
            gene_list.append(line.strip())
    filter_panelgenes(adata,gene_list)
    
    max_row = max_col = int((2000 / adata.uns["tissue_hires_scalef"]) + 1)
    spot_row = adata.obs["pxl_col_in_fullres"].values * (600 / max_row)
    spot_row = spot_row.astype(np.int)
    spot_col = adata.obs["pxl_row_in_fullres"].values * (600 / max_col)
    spot_col = spot_col.astype(np.int)
    radius = 4

    panel_gene = adata.X.A
    for thres in np.arange(1.0, 10.0, 1):
        thres_list = []
        for index in range(panel_gene.shape[0]):
            panel_gene[panel_gene < thres] = 0
            mo = 0
            for y in range(panel_gene.shape[1]):
                mo += panel_gene[index][y] * panel_gene[index][y]
            thres_list.append(sqrt(mo))

        # print(thres)
        image = np.zeros(shape=(600, 600, 3), dtype=np.uint8)
        for index in range(len(thres_list)):
            if thres_list[index] != 0:
                image[(spot_row[index] - radius):(spot_row[index] + radius),(spot_col[index] - radius):(spot_col[index] + radius)] = (255, 255, 255)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        cv2.imwrite(output_folder+'/'+str(thres) + '.png', image)


def parse_args():
    parser = argparse.ArgumentParser(description='category map segmentation')
    parser.add_argument('-matrix', type=str, nargs='+', help='h5 file path')
    parser.add_argument('-csv', type=str, nargs='+', help='metadata csv file path')
    parser.add_argument('-json', type=str, nargs='+', help='json file path')
    parser.add_argument('-out', '--output_path', type=str, nargs='*', default='output', help='generate output folder')
    parser.add_argument('-gene', type=str, nargs='+', help='panel gene txt  path,one line is a panel gene',default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    h5_path = args.matrix[0]
    spatial_path = args.csv[0]
    scale_factor_path = args.json[0]
    output_path = args.output_path[0]
    panel_gene_path = args.gene[0]

    expression_threshold_visulization(h5_path, spatial_path, scale_factor_path, output_path, panel_gene_path= panel_gene_path)


