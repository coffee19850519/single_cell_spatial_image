import os,csv,re,json
import pandas as pd
import numpy as np
import scanpy as sc
import warnings
import argparse
# from package_pipeline import pseudo_images
from package_pipeline_multiprocessing import  pseudo_images
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Generate RGB images')
    parser.add_argument('-expression', type=str, nargs='+', help='h5 file path')
    parser.add_argument('-meta', type=str, nargs='+', help='metadata csv file path')
    parser.add_argument('-scaler', type=str, nargs='+', help='json file path')
    parser.add_argument('-output', '--output_path', type=str, nargs='*', default=['output_img'], help='generate output folder')
    parser.add_argument('-embedding', type=str, nargs='+', default=['scGNN'], help='optional spaGCN or scGNN')
    parser.add_argument('-transform', type=str, nargs='+', default=['logcpm'], help='data transform optional is log or logcpm or None')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    h5_path = args.expression[0]
    spatial_path = args.meta[0]
    scale_factor_path = args.scaler[0]
    output_path = args.output_path[0]
    method = args.embedding[0]
    transform_opt = args.transform[0]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    pseudo_images(h5_path, spatial_path, scale_factor_path, output_path, method, None,False,transform_opt)

