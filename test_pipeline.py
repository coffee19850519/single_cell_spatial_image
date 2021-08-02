import os,csv,re,json
import pandas as pd
import numpy as np
import scanpy as sc
import warnings
import argparse
import time, resource
import csv
# from package_pipeline import segmentation_test
from package_pipeline_multiprocessing import  segmentation_test
warnings.filterwarnings("ignore")

def parse_args1():
    parser = argparse.ArgumentParser(description='Predict tissue architecture without annotation')
    parser.add_argument('-expression', type=str, nargs='+', help='h5 file path')
    parser.add_argument('-meta', type=str, nargs='+', help='metadata csv file path')
    parser.add_argument('-scaler', type=str, nargs='+', help='json file path')
    parser.add_argument('-output', '--output_path', type=str, nargs='*', default='output', help='generate output folder')
    parser.add_argument('-embedding', type=str, nargs='+', default=['scGNN'], help='optional spaGCN or scGNN')
    parser.add_argument('-transform', type=str, nargs='+', default=['None'], help='data transform optional is log or logcpm or None')
    parser.add_argument('-model', type=str, nargs='+', help='checkpoint path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args1 = parse_args1()

    h5_path = args1.expression[0]
    spatial_path = args1.meta[0]
    scale_factor_path = args1.scaler[0]
    output_path = args1.output_path[0]
    method = args1.embedding[0]
    transform_opt = args1.transform[0]
    checkpoint = args1.model[0]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    segmentation_test(h5_path, spatial_path, scale_factor_path, output_path, method,None,False,transform_opt,checkpoint)

