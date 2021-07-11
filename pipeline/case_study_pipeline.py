import os,csv,re,json
import pandas as pd
import numpy as np
import scanpy as sc
import warnings
import argparse
# from package_pipeline import pseudo_images
from package_pipeline_multiprocessing import  pseudo_images
from package_pipeline_multiprocessing import case_study_test

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='category map segmentation')
    parser.add_argument('-matrix', type=str, nargs='+', help='h5 file path')
    parser.add_argument('-csv', type=str, nargs='+', help='metadata csv file path')
    parser.add_argument('-json', type=str, nargs='+', help='json file path')
    parser.add_argument('-out', '--output_path', type=str, nargs='*', default='output', help='generate output folder')
    parser.add_argument('-gene', type=str, nargs='+', help='panel gene txt  path,one line is a panel gene',default=[None])
    parser.add_argument('-method', type=str, nargs='+', default=['scGNN'], help='optional spaGCN or scGNN')
    parser.add_argument('-pca', type=str, nargs='+', default=[True], help='pca optional:True or False')
    parser.add_argument('-transform', type=str, nargs='+', default=['None'], help='data transform optional is log or logcpm or None')
    parser.add_argument('-red_min', type=int, nargs='+', default=[(0,255)], help='red')
    parser.add_argument('-green_min', type=int, nargs='+', default=[(0,255)], help='green')
    parser.add_argument('-blue_min', type=int, nargs='+', default=[(0,255)], help='blue')
    parser.add_argument('-red_max', type=int, nargs='+', default=[(0,255)], help='red')
    parser.add_argument('-green_max', type=int, nargs='+', default=[(0,255)], help='green')
    parser.add_argument('-blue_max', type=int, nargs='+', default=[(0,255)], help='blue')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    h5_path = args.matrix[0]
    spatial_path = args.csv[0]
    scale_factor_path = args.json[0]
    output_path = args.output_path[0]
    panel_gene_path = args.gene[0]
    method = args.method[0]
    pca_opt = args.pca[0]
    transform_opt = args.transform[0]
    red_min = args.red_min[0]   
    green_min = args.green_min[0]   
    blue_min = args.blue_min[0]   
    red_max = args.red_max[0]   
    green_max = args.green_max[0]   
    blue_max = args.blue_max[0]   
    print(red_min,green_min,blue_min)
    print(red_max,green_max,blue_max)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    r_tuple = (red_min, red_max)
    g_tuple = (green_min, green_max)
    b_tuple = (blue_min, blue_max)
    case_study_test(h5_path, spatial_path, scale_factor_path, output_path, method,  panel_gene_path , pca_opt, transform_opt,r_tuple,g_tuple,b_tuple)


