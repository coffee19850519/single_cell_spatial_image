import pandas as pd
from pipeline_sparse_expression_to_image import save_transformed_RGB_to_image_and_csv, scale_to_RGB
import json, os
import numpy as np
import pickle as pkl


def transform_embedding_to_image(anndata,sample_name,img_folder, img_type, scale_factor_file= None):
    X_transform = anndata.obsm["embedding"]
    print(X_transform.shape)
    # embedding_data = pd.read_csv("LogCPM_151507_humanBrain_128_pca_0.2_res.csv")
    # X_transform = embedding_data.loc[:, ['embedding0', 'embedding1', 'embedding2']].values
    full_data = anndata.obs
    X_transform[:, 0] = scale_to_RGB(X_transform[:, 0], 100)
    X_transform[:, 1] = scale_to_RGB(X_transform[:, 1], 100)
    X_transform[:, 2] = scale_to_RGB(X_transform[:, 2], 100)

    if scale_factor_file is not None:   # uns

        radius = int(0.5 *  anndata.uns['fiducial_diameter_fullres'] + 1)
        # radius = int(scaler['spot_diameter_fullres'] + 1)
        max_row = max_col = int((2000 / anndata.uns['tissue_hires_scalef']) + 1)


    else:  # no 10x
        radius = 100 #
        max_row = np.int(np.max(full_data['pxl_col_in_fullres'].values + 1) + radius)   #
        max_col = np.int(np.max(full_data['pxl_row_in_fullres'].values + 1) + radius)

    high_img, low_img= save_transformed_RGB_to_image_and_csv(full_data['pxl_col_in_fullres'].values,
                                          full_data['pxl_row_in_fullres'].values,
                                          max_row,
                                          max_col,
                                          X_transform,
                                          sample_name,  # file name  default/define
                                          img_type,
                                          img_folder,
                                          plot_spot_radius = radius
                                          )

    del full_data, X_transform
    return high_img, low_img

