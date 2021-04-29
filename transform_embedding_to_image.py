import pandas as pd
from sparse_expression_to_image import save_transformed_RGB_to_image_and_csv, scale_to_RGB, correlation_between_enhanced_images,RGB_correlation_between_images
import json, os, cv2
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
            scaler =  json.load(fp_scaler)

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


def transform_category_to_image(result_file, meta_data_file, scale_factor_file= None,
                                ):
    meta_data = pd.read_csv(meta_data_file)

    result_data = pd.read_csv(result_file)
    # embedding_data.rename(columns={'Unnamed: 0': 'barcode'}, inplace=True)
    #merge them together
    full_data = pd.merge(meta_data,result_data, on = 'barcode')


    if scale_factor_file is not None:
        with open(scale_factor_file) as fp_scaler:
            scaler =  json.load(fp_scaler)

        plot_spot_radius = int(0.5 *  scaler['fiducial_diameter_fullres'] + 1)
        # radius = int(scaler['spot_diameter_fullres'] + 1)
        max_row = max_col = int((2000 / scaler['tissue_hires_scalef']) + 1)

    else:
        plot_spot_radius = 100 #
        max_row = np.int(np.max(full_data['pxl_col_in_fullres'].values + 1) + plot_spot_radius)
        max_col = np.int(np.max(full_data['pxl_row_in_fullres'].values + 1) + plot_spot_radius)

    img = np.ones(shape=(max_row + 1, max_col + 1, 3), dtype=np.uint8) * 255

    # color_dict = {'Layer1': (128, 9, 21),
    #               'Layer2': (50,14,77),
    #               'Layer3': (61,154,124),
    #               'Layer4': (59,170,246),
    #               'Layer5': (255, 255, 0),
    #              'Layer6': (255, 0, 255),
    #             'Layer7': (0, 255, 255),
    #               'WM':(255, 0, 0)}
    color_dict = {0: (128, 9, 21),
                  1: (50,14,77),
                  2: (61,154,124),
                  3: (59,170,246),
                  4: (255, 255, 0),
                  5: (255, 0, 255),
                  6: (0, 255, 255),
                  7:(255, 0, 0)}


    for index in range(len(full_data)):
        # img[spot_row[index], spot_col[index]] = values[index]
        # radius = 60
        cv2.rectangle(img,
                      (full_data['pxl_row_in_fullres_x'].values[index] - plot_spot_radius,
                       full_data['pxl_col_in_fullres_x'].values[index] - plot_spot_radius),
                      (full_data['pxl_row_in_fullres_x'].values[index] + plot_spot_radius,
                       full_data['pxl_col_in_fullres_x'].values[index] + plot_spot_radius),
                      color= color_dict[full_data['class_num'].values[index]],
                      thickness=-1)

        # cv2.circle(img, (spot_col_in_fullres[index], spot_row_in_fullres[index]), radius=plot_spot_radius,
        #            color=(int(X_transformed[index][0]), int(X_transformed[index][1]), int(X_transformed[index][2])),
        #            thickness=-1)

    hi_img = cv2.resize(img, dsize=(2000, 2000), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.splitext(result_file)[0] + '_transformed_hires.jpg', hi_img)
    low_img = cv2.resize(img, dsize=(600, 600), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.splitext(result_file)[0] + '_transformed_lowres.jpg', low_img)
    # save transformed_X back to csv file
    # X.insert(gene_start_idx, 'transformed_R',X_transformed[:, 0], True)
    # X.insert(gene_start_idx+1, 'transformed_G',X_transformed[:, 1], True)
    # X.insert(gene_start_idx+2, 'transformed_B',X_transformed[:, 2], True)
    #
    # # #then save to csv file
    # X.iloc[:, 0:gene_start_idx+3].to_csv(os.path.splitext(data_file_name)[0] + '_newRGB.csv', index = False)

    del img, hi_img, low_img  # , mask, inpaint_img,  resize_contour#resize_img,

    del full_data, meta_data,


if __name__ == "__main__":
    data_fold = r'/run/media/fei/Entertainment/pointAE/'

    all_data_names = []
    for file_name in os.listdir(data_fold):
        # data_name = os.path.splitext(file_name)[0].rsplit('_',1)[0]
        data_name = file_name.rsplit('_', 1)[0]
        if data_name not in all_data_names:
            all_data_names.append(data_name)

    from glob import glob

    for data_name in all_data_names:
        for full_name in glob(os.path.join(data_fold, data_name + '*_correlation.csv')):
            # embedding_file = os.path.join(data_fold, data_name + '_3dim_embed.csv')
            meta_data = os.path.join(data_fold, data_name + '_metaData.csv')
            scale_factor_file = os.path.join(data_fold, data_name + '_scaleFactors.json')
            # meta_data_file = r'/run/media/fei/Entertainment/151673_humanBrain_cpm_metaData.csv'
            # scale_factor_file = r'/run/media/fei/Entertainment/151673_humanBrain_cpm_scaleFactors.json'
            try:
                transform_embedding_to_image(embedding_file = full_name,
                                                      meta_data_file = meta_data,
                                                      scale_factor_file = scale_factor_file)
                # transform_category_to_image(os.path.join(data_fold, data_name + '_louvain.csv'), meta_data, scale_factor_file=scale_factor_file,
                #                             )
            except Exception as e:
                print(e)