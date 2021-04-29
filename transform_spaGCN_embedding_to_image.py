import pandas as pd
from sparse_expression_to_image import save_transformed_RGB_to_image_and_csv, scale_to_RGB, correlation_between_enhanced_images,RGB_correlation_between_images
import json, os
import numpy as np
import pickle as pkl


def transform_embedding_to_image(embedding_file, meta_data_file, scale_factor_file= None,
                                 original_RGB = False, max_iteration = 30):
    meta_data = pd.read_csv(meta_data_file)
    if os.path.splitext(embedding_file)[1] != '.csv':
        with open(embedding_file, 'rb') as f:
            X_transform = pkl.loads(f.read())
        full_data = meta_data.copy(True)
        full_data.insert(full_data.shape[1], 'embedding0', X_transform[:, 0])
        full_data.insert(full_data.shape[1], 'embedding1', X_transform[:, 1])
        full_data.insert(full_data.shape[1], 'embedding2', X_transform[:, 2])
    else:
        embedding_data = pd.read_csv(embedding_file)
        # embedding_data.rename(columns={'Unnamed: 0': 'barcode'}, inplace=True)
        #merge them together
        full_data = pd.merge(meta_data,embedding_data, on = 'barcode')
        X_transform = full_data.loc[:, ['embedding0','embedding1','embedding2']].values
        del embedding_data
    best_pcc = 0
    # best_pca_ratio = 0
    best_perm = (0, 1, 2)
    best_percentile_0 = 0
    best_percentile_1 = 0
    best_percentile_2 = 0
    if original_RGB:
        # save original image at spot level
        # save_spot_RGB_to_image(X, expression_file)
        #do grid search for the percentile parameters

        for percentile_0 in range(0, max_iteration):
            for percentile_1 in range(0, max_iteration):
                for percentile_2 in range(0, max_iteration):
                    # plot_box_for_RGB(X_transformed)
                    # optimize pseudo-RGB according to spatial image
                    temp_X_transform = X_transform.copy()
                    temp_X_transform[:, 0] = scale_to_RGB(X_transform[:, 0], percentile_0)
                    temp_X_transform[:, 1] = scale_to_RGB(X_transform[:, 1], percentile_1)
                    temp_X_transform[:, 2] = scale_to_RGB(X_transform[:, 2], percentile_2)
                    if 'equa' in full_data.columns:
                        pcc, perm = correlation_between_enhanced_images(full_data['equa'].values,
                                                                        full_data['array_row'].values,
                                                                        full_data['array_col'].values, temp_X_transform)
                    else:

                        pcc, perm = RGB_correlation_between_images(full_data[['R','G','B']].values, temp_X_transform)


                    if best_pcc < pcc:
                        best_pcc = pcc
                        # best_transformed_X = temp_X_transform[:, perm]
                        best_perm = perm
                        best_percentile_0 = percentile_0
                        best_percentile_1 = percentile_1
                        best_percentile_2 = percentile_2
                    del temp_X_transform

        #re-produce optimal RGB projection result
        X_transform[:, 0] = scale_to_RGB(X_transform[:, 0], best_percentile_0)
        X_transform[:, 1] = scale_to_RGB(X_transform[:, 1], best_percentile_1)
        X_transform[:, 2] = scale_to_RGB(X_transform[:, 2], best_percentile_2)

        print('the best pcc for the file {:s} is {:.3f}'.format(embedding_file,best_pcc))

    else:
        # directly rescale embeddings to RGB range
        X_transform[:, 0] = scale_to_RGB(X_transform[:, 0], 100)
        X_transform[:, 1] = scale_to_RGB(X_transform[:, 1], 100)
        X_transform[:, 2] = scale_to_RGB(X_transform[:, 2], 100)

    if scale_factor_file is not None:
        with open(scale_factor_file) as fp_scaler:
            scaler = json.load(fp_scaler)

        radius = int(0.5 *  scaler['fiducial_diameter_fullres'] + 1)
        # radius = int(scaler['spot_diameter_fullres'] + 1)
        max_row = max_col = int((2000 / scaler['tissue_hires_scalef']) + 1)


    else:
        radius = 100 #
        max_row = np.int(np.max(full_data['pxl_col_in_fullres'].values + 1) + radius)
        max_col = np.int(np.max(full_data['pxl_row_in_fullres'].values + 1) + radius)

    save_transformed_RGB_to_image_and_csv(full_data['pxl_col_in_fullres'].values,
                                          full_data['pxl_row_in_fullres'].values,
                                          max_row,
                                          max_col,
                                          X_transform[:, best_perm],
                                          embedding_file,
                                          plot_spot_radius = radius)

    del full_data, meta_data, X_transform


if __name__ == "__main__":
    data_fold = r'/run/media/fei/Entertainment/SpaGCN_4_embedding/'
    # norm_list = ['TMM', 'TPM', 'scTransform', 'DESeq2','FPKM','scran','LogCPM']
    norm_list = ['velocity']
    all_data_names = []
    for file_name in os.listdir( os.path.join(data_fold, 'meta')):
        # data_name = os.path.splitext(file_name)[0].rsplit('_',1)[0]
        sample_name = file_name.split('_',1)[0]

        if sample_name not in all_data_names:
            all_data_names.append(sample_name)

    from glob import glob

    for data_name in all_data_names:
        for norm in norm_list:
            for full_name in glob(os.path.join(data_fold, 'Velocity_embedding', norm + '_' + data_name + '_humanBrain_*.csv')):
                # embedding_file = os.path.join(data_fold, data_name + '_3dim_embed.csv')
                meta_data = os.path.join(data_fold, 'meta', data_name + '_humanBrain_metaData.csv')
                scale_factor_file = os.path.join(data_fold, 'scale', data_name + '_scalefactors.json')
                # meta_data_file = r'/run/media/fei/Entertainment/151673_humanBrain_cpm_metaData.csv'
                # scale_factor_file = r'/run/media/fei/Entertainment/151673_humanBrain_cpm_scaleFactors.json'
                try:
                    transform_embedding_to_image(embedding_file = full_name,
                                                          meta_data_file = meta_data,
                                                          scale_factor_file = scale_factor_file)
                except Exception as e:
                    print(e)