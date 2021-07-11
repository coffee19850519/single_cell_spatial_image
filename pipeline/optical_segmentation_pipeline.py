import os,csv,re,json
import pandas as pd
import numpy as np
import scanpy as sc
import warnings
import argparse
from package_pipeline import segmentation_category_map

warnings.filterwarnings("ignore")

def parse_args1():
    parser = argparse.ArgumentParser(description='category map segmentation')
    parser.add_argument('-matrix', type=str, nargs='+', help='h5 file path')
    parser.add_argument('-csv', type=str, nargs='+', help='metadata csv file path')
    parser.add_argument('-json', type=str, nargs='+', help='json file path')
    parser.add_argument('-optical', type=str, nargs='+', help='optical image path')  
    # parser.add_argument('-optical', type=str, nargs='+', help='optical image path')
    parser.add_argument('-out', '--output_path', type=str, nargs='*', default='output', help='generate output folder')
    parser.add_argument('-pca', type=str, nargs='+', default=[True], help='pca optional:True or False')
    parser.add_argument('-transform', type=str, nargs='+', default=['None'], help='data transform optional is log or logcpm or None')
    parser.add_argument('-checkpoint', type=str, nargs='+', help='checkpoint path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args1 = parse_args1()

    h5_path = args1.matrix
    spatial_path = args1.csv
    scale_factor_path = args1.json
    optical_path = args1.optical
    output_path = args1.output_path
    method = args1.method[0]
    pca_opt = args1.pca[0]
    transform_opt = args1.transform[0]
    checkpoint = args1.checkpoint[0]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # print('ok')
    segmentation_category_map(h5_path[0], spatial_path[0], scale_factor_path[0], optical_path[0], output_path[0],method,pca_opt,transform_opt,checkpoint)


# def seg_category_map(h5_path, spatial_path, scale_factor_path, optical_path, img_output_path, seg_output_path):
#     optical_img = cv2.imread(optical_path)
#     category_map = segmentation_test(h5_path, spatial_path, scale_factor_path, img_output_path)
#     seg_category_map(optical_img, category_map, seg_output_path)
