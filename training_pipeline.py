import os,csv,re,json
import pandas as pd
import numpy as np
import scanpy as sc
import warnings
import argparse
# from package_pipeline import pseudo_images
from package_pipeline_multiprocessing import pseudo_images
from package_pipeline_multiprocessing import case_study_test
from PIL import Image
from train import train
import cv2
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('-data_folder', type=str, nargs='+', help='a folder provides all training samples.The data including label file of each sample should follow our predefined schema in a sub-folder under this folder.')
    parser.add_argument('-model', type=str, nargs='+',default=[None], help='file path for pre-trained model file')
    parser.add_argument('-output', type=str, nargs='*', default='output', help='output root folder')
    parser.add_argument('-embedding', type=str, nargs='+', default=['scGNN'], help='optional spaGCN or scGNN')
    parser.add_argument('-transform', type=str, nargs='+', default=['None'], help='data transform optional is log or logcpm or None')


    args = parser.parse_args()
    return args

def save_spot_RGB_to_image(label_path, adata):
    X = pd.read_csv(label_path)
    X = X.sort_values(by=['barcode'])
    assert all(adata.obs.index == X.iloc[:, 0].values)
    layers = X.iloc[:, 1].values
    spot_row = adata.obs["pxl_col_in_fullres"]
    spot_col = adata.obs["pxl_row_in_fullres"]
    radius = int(0.5 * adata.uns['fiducial_diameter_fullres'] + 1)
    max_row = max_col = int((2000 / adata.uns['tissue_hires_scalef']) + 1)
    img = np.zeros(shape=(max_row + 1, max_col + 1), dtype=np.int)

    img = img.astype(np.uint8)
    for index in range(len(layers)):
        if layers[index] == 'Layer1':
            img[(spot_row[index] - radius):(spot_row[index] + radius),
            (spot_col[index] - radius):(spot_col[index] + radius)] = 1
        elif layers[index] == 'Layer2':
            img[(spot_row[index] - radius):(spot_row[index] + radius),
            (spot_col[index] - radius):(spot_col[index] + radius)] = 2
        elif layers[index] == 'Layer3':
            img[(spot_row[index] - radius):(spot_row[index] + radius),
            (spot_col[index] - radius):(spot_col[index] + radius)] = 3
        elif layers[index] == 'Layer4':
            img[(spot_row[index] - radius):(spot_row[index] + radius),
            (spot_col[index] - radius):(spot_col[index] + radius)] = 4
        elif layers[index] == 'Layer5':
            img[(spot_row[index] - radius):(spot_row[index] + radius),
            (spot_col[index] - radius):(spot_col[index] + radius)] = 5
        elif layers[index] == 'Layer6':
            img[(spot_row[index] - radius):(spot_row[index] + radius),
            (spot_col[index] - radius):(spot_col[index] + radius)] = 6
        elif layers[index] == 'WM':
            img[(spot_row[index] - radius):(spot_row[index] + radius),
            (spot_col[index] - radius):(spot_col[index] + radius)] = 7

    shape = adata.uns["img_shape"]
    label_img = cv2.resize(img, dsize=(shape, shape), interpolation=cv2.INTER_NEAREST)
    return label_img



def train_preprocessing(path, sample_name, adata, output_folder):

    output_path = './'+output_folder+'/RGB_images_label/'
    img_path = './'+output_folder+'/RGB_images/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for name in os.listdir(img_path):
        sample = name.split('_',4)[0]
        if sample_name == sample:

            data_file = path+'/'+sample+'/'+sample+'_annotation.csv'
            label = save_spot_RGB_to_image(data_file, adata)
            savehi =  Image.fromarray(label)
            savehi.convert("P").save(os.path.join(output_path,name))


def load_data(h5_path, spatial_path, scale_factor_path):
    # Read in gene expression and spatial location
    adata = sc.read_10x_h5(h5_path)
    spatial_all = pd.read_csv(spatial_path, sep=",", header=None, na_filter=False, index_col=0)
    spatial = spatial_all[spatial_all[1] == 1]
    spatial = spatial.sort_values(by=0)
    assert all(adata.obs.index == spatial.index)
    adata.obs["in_tissue"] = spatial[1]
    adata.obs["array_row"] = spatial[2]
    adata.obs["array_col"] = spatial[3]
    adata.obs["pxl_col_in_fullres"] = spatial[4]
    adata.obs["pxl_row_in_fullres"] = spatial[5]
    adata.obs.index.name = 'barcode'
    adata.var_names_make_unique()

    # Read scale_factor_file
    with open(scale_factor_path) as fp_scaler:
        scaler = json.load(fp_scaler)
    adata.uns["spot_diameter_fullres"] = scaler["spot_diameter_fullres"]
    adata.uns["tissue_hires_scalef"] = scaler["tissue_hires_scalef"]
    adata.uns["fiducial_diameter_fullres"] = scaler["fiducial_diameter_fullres"]
    adata.uns["tissue_lowres_scalef"] = scaler["tissue_lowres_scalef"]

    return adata , spatial_all


if __name__ == '__main__':

    args = parse_args()
    path = args.data_folder[0]
    model = args.model[0]
    output_folder = args.output[0]
    method = args.embedding[0]
    transform_opt = args.transform[0]
    for name in os.listdir(path):
        h5_path = path+'/'+name+'/filtered_feature_bc_matrix.h5'
        spatial_path = path +'/'+name+'/spatial/tissue_positions_list.csv'
        scale_factor_path = path +'/'+name+'/spatial/scalefactors_json.json'
        adata,spatial_all = load_data(h5_path, spatial_path, scale_factor_path)
        adata.uns["img_shape"] = 600

        pseudo_images(h5_path, spatial_path, scale_factor_path, output_folder,method, None, False, transform_opt)
        train_preprocessing(path, name, adata, output_folder)
    config = './configs/config.py'
    train(config, model, output_folder)


