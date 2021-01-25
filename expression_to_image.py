import pandas as pd
import umap
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from itertools import permutations

def scale_to_RGB(channel,truncated_percent):
    truncated_down = np.percentile(channel, truncated_percent)
    truncated_up = np.percentile(channel, 100 - truncated_percent)
    channel_new = ((channel - truncated_down) / (truncated_up - truncated_down)) * 255
    channel_new[channel_new < 0] = 0
    channel_new[channel_new > 255] = 255
    return np.uint8(channel_new)

def correlation_between_images(X_RGB, transformed_RGB):
    best_pcc = 0
    best_perm = (0,1,2)
    for perm in permutations(range(3), 3):
        current_pcc, p_value = pearsonr(X_RGB.reshape((-1)), transformed_RGB[:, perm].reshape((-1)))
        if abs(current_pcc) > best_pcc:
            best_pcc = abs(current_pcc)
            best_perm = perm
    return  best_pcc, best_perm

def nan_check_in_datframe():
    pass




'''
expression_file: file path for processed expression matrix
pca_conponent_num (50 in default): number of principle components remains in PCA denoising
umap_neighbor_num (10 in default): the size of the local neighborhood in UMAP
umap_min_dist (0.2 in default): control how tightly UMAP is allowed to pack points

'''
def transform_expression_to_RGB(expression_file, pca_conponent_num = 50, umap_neighbor_num = 10, umap_min_dist = 0.2 ):
    data = pd.read_csv(expression_file)

    X = data.loc[data['in_tissue'] != 0]

    # read RGB columns to see if needing RGB optimization
    if np.sum(data.loc[:, ['R','G','B']].values) != 0:
        original_RGB = False
    else:
        original_RGB = True
        # save original image at spot level
        save_spot_RGB_to_image(X, expression_file)


    if 'equa' in data.columns:
        #set gene start idex
        gene_start_idx = 10
        #save enhanced image at spot level
        save_enhanced_RGB_to_image(X, expression_file)
    else:
        gene_start_idx = 9

    # save
    values = X.iloc[:, gene_start_idx:].values

    best_pcc = 0
    # best_pca_ratio = 0
    best_perm = (0,1,2)
    best_percentile_0 = 0
    best_percentile_1 = 0
    best_percentile_2 = 0

    # for pca_ratio in np.arange(0.75, 1.1, 0.01):
        # apply PCA to denoise first
    pac_model = PCA(n_components=pca_conponent_num, svd_solver = 'full')
    transformer = umap.UMAP(n_neighbors=umap_neighbor_num, min_dist=umap_min_dist, n_components=3)
    try:
        values = pac_model.fit_transform(values)
        X_transformed = transformer.fit(values).transform(values)
    except Exception as e:
        print(str(e))
        return
    if original_RGB:
        #do grid search for the percentile parameters
        for percentile_0 in range(0, 50):
            for percentile_1 in range(0, 50):
                for percentile_2 in range(0, 50):
                    # plot_box_for_RGB(X_transformed)
                    temp_X_transform = np.zeros_like(X_transformed, np.int8)
                    temp_X_transform[:, 0] = scale_to_RGB(X_transformed[:, 0], percentile_0)
                    temp_X_transform[:, 1] = scale_to_RGB(X_transformed[:, 1], percentile_1)
                    temp_X_transform[:, 2] = scale_to_RGB(X_transformed[:, 2], percentile_2)

                    pcc, perm = correlation_between_images(X.iloc[:, 6:9].values, temp_X_transform)
                    if best_pcc < pcc:
                        best_pcc = pcc
                        # best_transformed_X = temp_X_transform[:, perm]
                        best_perm = perm
                        best_percentile_0 = percentile_0
                        best_percentile_1 = percentile_1
                        best_percentile_2 = percentile_2
                    del temp_X_transform

        #re-produce optimal RGB projection result
        X_transformed[:, 0] = scale_to_RGB(X_transformed[:, 0], best_percentile_0)
        X_transformed[:, 1] = scale_to_RGB(X_transformed[:, 1], best_percentile_1)
        X_transformed[:, 2] = scale_to_RGB(X_transformed[:, 2], best_percentile_2)

        print('the best pcc for the file {:s} is {:.3f}'.format(expression_file,best_pcc))

    else:
        # directly rescale embeddings to RGB range
        X_transformed[:, 0] = scale_to_RGB(X_transformed[:, 0], 100)
        X_transformed[:, 1] = scale_to_RGB(X_transformed[:, 1], 100)
        X_transformed[:, 2] = scale_to_RGB(X_transformed[:, 2], 100)

    save_transformed_RGB_to_image_and_csv(X,X_transformed[:, best_perm], gene_start_idx, expression_file)

    del X_transformed, data, X, values, transformer





def plot_box_for_RGB(RGB_data):
    fig = plt.figure()
    plt.boxplot(RGB_data)
    plt.xticks([1,2,3], ['R\'','G\'','B\''],  rotation=0)
    plt.show()

def save_transformed_RGB_to_image_and_csv(X, X_transformed, gene_start_idx ,data_file_name):

    # full size spot number
    spot_row = X.loc[:, 'pxl_col_in_fullres'].values
    spot_col = X.loc[:, 'pxl_row_in_fullres'].values

    max_row = np.int(np.max(spot_row + 1))
    max_col = np.int(np.max(spot_col + 1))

    img = np.ones(shape=(max_row + 1, max_col + 1, 3), dtype=np.uint8) * 255

    for index in range(len(X_transformed)):
        # img[spot_row[index], spot_col[index]] = values[index]
        # radius = 60
        # cv2.rectangle(img, (spot_col[index] - radius-7, spot_row[index] - radius),
        #               (spot_col[index] + radius+7, spot_row[index]+ radius),
        #               color=(int(values[index][0]), int(values[index][1]), int(values[index][2])),
        #               thickness=-1)
        cv2.circle(img, (spot_col[index], spot_row[index]), radius=68,
                   color=(int(X_transformed[index][0]), int(X_transformed[index][1]), int(X_transformed[index][2])),
                   thickness=-1)



    cv2.imwrite(os.path.splitext(data_file_name)[0] + '_transformed.jpg', img)

    # save transformed_X back to csv file
    X.insert(gene_start_idx, 'transformed_R',X_transformed[:, 0], True)
    X.insert(gene_start_idx+1, 'transformed_G',X_transformed[:, 1], True)
    X.insert(gene_start_idx+2, 'transformed_B',X_transformed[:, 2], True)

    # #then save to csv file
    X.iloc[:, 0:gene_start_idx+3].to_csv(os.path.splitext(data_file_name)[0] + '_newRGB.csv', index = False)

    del img, spot_row, spot_col#, mask, inpaint_img,  resize_contour#resize_img,

def save_pseudoRGB_to_spot_level(X, gene_start_idx, data_path):
    values = X.iloc[:, gene_start_idx:].values
    spot_row  = X.loc[:, 'pxl_col_in_fullres'].values
    spot_col = X.loc[:, 'pxl_row_in_fullres'].values
    max_row = np.int(np.max(spot_row + 1))
    max_col = np.int(np.max(spot_col + 1))
    img = np.ones(shape=(max_row + 1, max_col + 1, 3), dtype=np.uint8) * 255
    for index in range(len(values)):
        cv2.circle(img, (spot_col[index], spot_row[index]), radius= 68,
                   color=(int(values[index][0]), int(values[index][1]), int(values[index][2])),
                   thickness=-1)
    resize_img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.splitext(data_path)[0] + '_spot_rect.jpg', resize_img)
    del values, spot_row, max_col, img

def save_spot_RGB_to_image(X, data_file ):

    values = X[['R','G','B']].values
    spot_row = X.iloc[:, 'pxl_col_in_fullres'].values
    spot_col = X.iloc[:, 'pxl_row_in_fullres'].values

    max_row = np.int(np.max(spot_row))
    max_col = np.int(np.max(spot_col))

    img = np.ones(shape=(max_row+1, max_col+1,  3), dtype= np.float) * 255

    for index in range(len(values)):
        img[spot_row[index], spot_col[index]] = values[index]

    resize_img = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.splitext(data_file)[0] + '_raw.jpg', resize_img)
    del values, spot_row, max_col, img, resize_img

def save_enhanced_RGB_to_image(X, data_file):

    values = X[:,'equa'].values
    spot_row = X.iloc[:, 'pxl_col_in_fullres'].values
    spot_col = X.iloc[:, 'pxl_row_in_fullres'].values

    max_row = np.int(np.max(spot_row))
    max_col = np.int(np.max(spot_col))

    img = np.ones(shape=(max_row+1, max_col+1,  1), dtype= np.float) * 255

    for index in range(len(values)):
        img[spot_row[index], spot_col[index]] = values[index]

    resize_img = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.splitext(data_file)[0] + '_enhanced.jpg', resize_img)
    del values, spot_row, max_col, img, resize_img


#
data_folder = r'/home/fei/Desktop/test/Brain/'
for expression_file in os.listdir(data_folder):
    if os.path.splitext(expression_file)[1] == '.csv':
        transform_expression_to_RGB(os.path.join(data_folder, expression_file))



