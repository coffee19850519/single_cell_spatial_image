import os,csv,re,json
import pandas as pd
import numpy as np
import scanpy as sc
import warnings
import argparse
import time, resource
import csv
# from package_pipeline import segmentation_test
from package_pipeline_multiprocessing import  segmentation_evaluation
warnings.filterwarnings("ignore")

def parse_args1():
    parser = argparse.ArgumentParser(description='segmentation test')
    parser.add_argument('-expression', type=str, nargs='+', help='h5 file path')
    parser.add_argument('-meta', type=str, nargs='+', help='metadata csv file path')
    parser.add_argument('-scaler', type=str, nargs='+', help='json file path')
    parser.add_argument('-label', '--label_path', type=str, nargs='*',  help='label folder')
    parser.add_argument('-output', '--output_path', type=str, nargs='*', default='output', help='generate output folder')
    parser.add_argument('-gene', type=str, nargs='+', help='panel gene txt  path,one line is a panel gene',
                        default=[None])
    parser.add_argument('-embedding', type=str, nargs='+', default=['scGNN'], help='optional spaGCN or scGNN')
    parser.add_argument('-pca', type=str, nargs='+', default=[True], help='pca optional:True or False')
    parser.add_argument('-transform', type=str, nargs='+', default=['None'], help='data transform optional is log or logcpm or None')
    parser.add_argument('-model', type=str, nargs='+', help='checkpoint path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args1 = parse_args1()

    h5_path = args1.expression[0]
    spatial_path = args1.meta[0]
    scale_factor_path = args1.scaler[0]
    label_path = args1.label_path[0]
    output_path = args1.output_path[0]
    panel_gene_path = 'None'
    method = args1.embedding[0]
    pca_opt = args1.pca[0]
    transform_opt = args1.transform[0]
    checkpoint = args1.model[0]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # segmentation_category_map(h5_path[0], spatial_path[0], scale_factor_path[0], optical_path[0], output_path)
    segmentation_evaluation(h5_path, spatial_path, scale_factor_path, output_path, method,label_path,panel_gene_path,pca_opt,transform_opt,checkpoint)

