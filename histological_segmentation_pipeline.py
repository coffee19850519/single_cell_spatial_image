import os,csv,re,json
import pandas as pd
import numpy as np
import scanpy as sc
import warnings
import argparse
from package_pipeline_multiprocessing import segmentation_category_map

warnings.filterwarnings("ignore")

def parse_args1():
    parser = argparse.ArgumentParser(description='category map segmentation')
    parser.add_argument('-expression', type=str, nargs='+', help='h5 file path')
    parser.add_argument('-meta', type=str, nargs='+', help='metadata csv file path')
    parser.add_argument('-scaler', type=str, nargs='+', help='json file path')
    parser.add_argument('-model', type=str, nargs='+',default=[None], help='model path')
    parser.add_argument('-histological', type=str, nargs='+', help=' histological image path')  
    parser.add_argument('-output',  type=str, nargs='*', default=['output_histological'], help='generate output folder')
    parser.add_argument('-embedding', type=str, nargs='+', default=['spaGCN'], help='optional spaGCN or scGNN')
    parser.add_argument('-transform', type=str, nargs='+', default=['None'], help='data transform optional is log or logcpm or None')


    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args1 = parse_args1()

    h5_path = args1.expression[0]
    spatial_path = args1.meta[0]
    scale_factor_path = args1.scaler[0]
    optical_path = args1.histological[0]
    output_path = args1.output[0]
    model = args1.model[0]
    method = args1.embedding[0]
    transform_opt = args1.transform[0]

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    segmentation_category_map(h5_path, spatial_path, scale_factor_path, optical_path, output_path, method, None, False, transform_opt, model)

