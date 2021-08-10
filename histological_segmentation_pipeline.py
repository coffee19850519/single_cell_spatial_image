import os,csv,re,json
import pandas as pd
import numpy as np
import scanpy as sc
import warnings
import argparse
from package_pipeline_multiprocessing import segmentation_category_map

warnings.filterwarnings("ignore")

def parse_args1():
    parser = argparse.ArgumentParser(description='segment a histological image')
    parser.add_argument('-expression', type=str, nargs='+', help='file path for raw gene expression data')
    parser.add_argument('-meta', type=str, nargs='+', help='file path for spatial meta data recording tissue positions')
    parser.add_argument('-scaler', type=str, nargs='+', help='file path for scale factors')
    parser.add_argument('-k', type=int, nargs='+', default=[7], help='the number of tissue architectures')
    parser.add_argument('-model', type=str, nargs='+', help='file path for pre-trained model')
    parser.add_argument('-histological', type=str, nargs='+', help='file path for the corresponding histological image')
    parser.add_argument('-output',  type=str, nargs='+', help='generate output folder')
    parser.add_argument('-embedding', type=str, nargs='+', default=['spaGCN'], help='embedding method in use: scGNN or spaGCN')
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
    optical_path = args1.histological[0]
    output_path = args1.output[0]
    model = args1.model[0]
    method = args1.embedding[0]
    transform_opt = args1.transform[0]
    device = args1.device[0]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    segmentation_category_map(h5_path, spatial_path, scale_factor_path, optical_path, output_path, method, None, False, transform_opt, model, device, k)

