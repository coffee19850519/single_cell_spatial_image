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
    parser = argparse.ArgumentParser(description='category map segmentation')
    parser.add_argument('-data', type=str, nargs='+', help='h5, csv and json file path')
    parser.add_argument('-config', type=str, nargs='+', help='config path')
    parser.add_argument('-model', type=str, nargs='+',default=[None], help='model path')
    parser.add_argument('-out', '--output_path', type=str, nargs='*', default='output', help='generate output folder')
    parser.add_argument('-gene', type=str, nargs='+', help='panel gene txt  path,one line is a panel gene',default=[None])
    parser.add_argument('-method', type=str, nargs='+', default=['scGNN'], help='optional spaGCN or scGNN')
    parser.add_argument('-pca', type=str, nargs='+', default=[True], help='pca optional:True or False')
    parser.add_argument('-transform', type=str, nargs='+', default=['None'], help='data transform optional is log or logcpm or None')


    args = parser.parse_args()
    return args

def save_spot_RGB_to_image(data_file, scale_factor_file,h=600):
    # data_file = os.path.join(data_folder, expression_file)
    X = pd.read_csv(data_file)

    layers = X.iloc[:,9].values

    spot_row = X.iloc[:,4].values
    spot_col = X.iloc[:,5].values

    if scale_factor_file is not None:
        with open(scale_factor_file) as fp_scaler:
            scaler =  json.load(fp_scaler)

        radius = int(0.5 *  scaler['fiducial_diameter_fullres'] + 1)
        # radius = int(scaler['spot_diameter_fullres'] + 1)
        max_row = max_col = int((2000 / scaler['tissue_hires_scalef']) + 1)


    else:
        radius = 100 #
        max_row = np.int(np.max(meta_data['pxl_col_in_fullres'].values + 1) + radius)
        max_col = np.int(np.max(meta_data['pxl_row_in_fullres'].values + 1) + radius)
    # max_row = np.max(spot_row)
    # max_col = np.max(spot_col)

    img = np.zeros(shape=(max_row+1, max_col+1), dtype= np.int)
    # print(img[0,0])
    # print(len(layers))
    # print('&&&&&&&&',img)
    img=img.astype(np.uint8)
    for index in range(len(layers)):
        # if layers[index] =='Layer 1':
        #     # img[spot_row[index], spot_col[index]] = [0,0,255]
        #     img[(spot_row[index] - radius):(spot_row[index] + radius),(spot_col[index] - radius):(spot_col[index] + radius)] = 1
        #     # print(img[spot_row[index],spot_col[index]])
        #     # cv2.circle(img,(spot_row[index], spot_col[index]),radius,(0,0,255),thickness=-1)
        # elif layers[index] =='Layer 2':
        #     img[(spot_row[index] - radius):(spot_row[index] + radius),(spot_col[index] - radius):(spot_col[index] + radius)] = 2
        #     # img[spot_row[index], spot_col[index]] = [0,255,0]
        #     # cv2.circle(img,(spot_row[index], spot_col[index]),radius,(0,255,0),thickness=-1)
        #     # print(img[spot_row[index],spot_col[index]])
        # elif layers[index] =='Layer 3':
        #     img[(spot_row[index] - radius):(spot_row[index] + radius),(spot_col[index] - radius):(spot_col[index] + radius)] = 3
        #     # img[spot_row[index], spot_col[index]] = [255,0,0]
        #     # cv2.circle(img,(spot_row[index], spot_col[index]),radius,(255,0,0),thickness=-1)
        # elif layers[index] =='Layer 4':
        #     img[(spot_row[index] - radius):(spot_row[index] + radius),(spot_col[index] - radius):(spot_col[index] + radius)] = 4
        #     # img[spot_row[index], spot_col[index]] = [255,0,255]
        #     # cv2.circle(img,(spot_row[index], spot_col[index]),radius,(255,0,255),thickness=-1)
        # elif layers[index] =='Layer 5':
        #     img[(spot_row[index] - radius):(spot_row[index] + radius),(spot_col[index] - radius):(spot_col[index] + radius)] = 5
        #     # img[spot_row[index], spot_col[index]] = [0,255,255]
        #     # cv2.circle(img,(spot_row[index], spot_col[index]),radius,(0,255,255),thickness=-1)
        # elif layers[index] =='Layer 6':
        #     img[(spot_row[index] - radius):(spot_row[index] + radius),(spot_col[index] - radius):(spot_col[index] + radius)] = 6
        #     # img[spot_row[index], spot_col[index]] = [255,255,0]
        #     # cv2.circle(img,(spot_row[index], spot_col[index]),radius,(255,255,0),thickness=-1)
        # elif layers[index] =='White matter':
        #     img[(spot_row[index] - radius):(spot_row[index] + radius),(spot_col[index] - radius):(spot_col[index] + radius)] = 7
        #     # img[spot_row[index], spot_col[index]] = [0,0,0]
        #     cv2.circle(img,(spot_row[index], spot_col[index]),radius,(0,0,0),thickness=-1)

        if layers[index] =='Layer1':
            # img[spot_row[index], spot_col[index]] = [0,0,255]
            img[(spot_row[index] - radius):(spot_row[index] + radius),(spot_col[index] - radius):(spot_col[index] + radius)] = 1
            # print(img[spot_row[index],spot_col[index]])
            # cv2.circle(img,(spot_row[index], spot_col[index]),radius,(0,0,255),thickness=-1)
        elif layers[index] =='Layer2':
            img[(spot_row[index] - radius):(spot_row[index] + radius),(spot_col[index] - radius):(spot_col[index] + radius)] = 2
            # img[spot_row[index], spot_col[index]] = [0,255,0]
            # cv2.circle(img,(spot_row[index], spot_col[index]),radius,(0,255,0),thickness=-1)
            # print(img[spot_row[index],spot_col[index]])
        elif layers[index] =='Layer3':
            img[(spot_row[index] - radius):(spot_row[index] + radius),(spot_col[index] - radius):(spot_col[index] + radius)] = 3
            # img[spot_row[index], spot_col[index]] = [255,0,0]
            # cv2.circle(img,(spot_row[index], spot_col[index]),radius,(255,0,0),thickness=-1)
        elif layers[index] =='Layer4':
            img[(spot_row[index] - radius):(spot_row[index] + radius),(spot_col[index] - radius):(spot_col[index] + radius)] = 4
            # img[spot_row[index], spot_col[index]] = [255,0,255]
            # cv2.circle(img,(spot_row[index], spot_col[index]),radius,(255,0,255),thickness=-1)
        elif layers[index] =='Layer5':
            img[(spot_row[index] - radius):(spot_row[index] + radius),(spot_col[index] - radius):(spot_col[index] + radius)] = 5
            # img[spot_row[index], spot_col[index]] = [0,255,255]
            # cv2.circle(img,(spot_row[index], spot_col[index]),radius,(0,255,255),thickness=-1)
        elif layers[index] =='Layer6':
            img[(spot_row[index] - radius):(spot_row[index] + radius),(spot_col[index] - radius):(spot_col[index] + radius)] = 6
            # img[spot_row[index], spot_col[index]] = [255,255,0]
            # cv2.circle(img,(spot_row[index], spot_col[index]),radius,(255,255,0),thickness=-1)
        elif layers[index] =='WM':
            img[(spot_row[index] - radius):(spot_row[index] + radius),(spot_col[index] - radius):(spot_col[index] + radius)] = 7
            # img[spot_row[index], spot_col[index]] = [0,0,0]
            # cv2.circle(img,(spot_row[index], spot_col[index]),radius,(0,0,0),thickness=-1)



    hi_img = cv2.resize(img, dsize= (h,h),  interpolation= cv2.INTER_NEAREST)
    return hi_img



def train_preprocessing(path):

    output_path = './pseudo_images_label'
    img_path = './pseudo_images/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for name in os.listdir(img_path):
        sample = name.split('_',4)[0]
        # openname = sample+'.png'
        # data_file = path+sample+'_tissue_positions_list.csv'
        data_file = path+sample+'.csv'
        scale_factor_file = path+sample+'_scalefactors_json.json'
        label = save_spot_RGB_to_image(data_file, scale_factor_file,h=600)
        savehi =  Image.fromarray(label)
        savehi.convert("P").save(os.path.join(output_path,name))





if __name__ == '__main__':

    args = parse_args()
    path = args.data[0]
    config = args.config[0]
    model = args.model[0]
    output_folder = './'
    panel_gene_path = args.gene[0]
    method = args.method[0]
    pca_opt = args.pca[0]
    transform_opt = args.transform[0]
    if not os.path.exists('./pseudo_images/'):
        os.makedirs('./pseudo_images/')
    for name in os.listdir(path):
        # print(name)
        if name.split('.',3)[-1]=='h5':
            print(name)
            sample = name.split('_',6)[0]
            h5_path = path+sample+'_filtered_feature_bc_matrix.h5'
            spatial_path = path+sample+'_tissue_positions_list.csv'
            scale_factor_path = path+sample+'_scalefactors_json.json'
            pseudo_images(h5_path, spatial_path, scale_factor_path, output_folder,method, panel_gene_path, pca_opt, transform_opt)

    # path = './pseudo_images/pseudo_images/'
    
    train_preprocessing(path)
    train(config, model)


