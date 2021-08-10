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
    parser = argparse.ArgumentParser(description='evaluate predictive tissue architecture with annotations')
    parser.add_argument('-expression', type=str, nargs='+', help='file path for raw gene expression data')
    parser.add_argument('-meta', type=str, nargs='+', help='file path for spatial meta data recording tissue positions')
    parser.add_argument('-scaler', type=str, nargs='+', help='file path for scale factors')
    parser.add_argument('-k', type=int, nargs='+', default=[7], help='the number of tissue architectures')
    parser.add_argument('-label', '--label_path', type=str, nargs='+',  help='file path for labels recording spot barcodes and their annotations for calculating evaluation metrics')
    parser.add_argument('-model', type=str, nargs='+', help='file path for pre-trained model')
    parser.add_argument('-output', '--output_path', type=str, nargs='+', help='output root folder')
    parser.add_argument('-embedding', type=str, nargs='+', default=['scGNN'], help='embedding method in use: scGNN or spaGCN')
    parser.add_argument('-transform', type=str, nargs='+', default=['logcpm'], help='data pre-transform method: log, logcpm or None')
    parser.add_argument('-device', type=str, nargs='+', default=['cpu'], help='cpu/gpu device option: cpu or gpu')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args1 = parse_args1()

    h5_path = args1.expression[0]
    spatial_path = args1.meta[0]
    scale_factor_path = args1.scaler[0]
    k = args1.k[0]
    label_path = args1.label_path[0]
    output_path = args1.output_path[0]
    method = args1.embedding[0]
    transform_opt = args1.transform[0]
    checkpoint = args1.model[0]
    device = args1.device[0]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    segmentation_evaluation(h5_path, spatial_path, scale_factor_path, output_path, method,label_path,None,False,transform_opt,checkpoint, device, k)

