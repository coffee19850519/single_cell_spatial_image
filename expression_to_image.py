import pandas as pd
import umap
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import IncrementalPCA
from itertools import permutations

def scale_to_RGB(channel,truncated_percent):
    truncated_down = np.percentile(channel, truncated_percent)
    truncated_up = np.percentile(channel, 100 - truncated_percent)
    channel_new = ((channel - truncated_down) / (truncated_up - truncated_down)) * 255
    channel_new[channel_new < 0] = 0
    channel_new[channel_new > 255] = 255
    return np.uint8(channel_new)

def RGB_correlation_between_images(X_RGB, transformed_RGB):
    best_pcc = 0
    best_perm = (0,1,2)
    for perm in permutations(range(3), 3):
        current_pcc, p_value = pearsonr(X_RGB.reshape((-1)), transformed_RGB[:, perm].reshape((-1)))
        if abs(current_pcc) > best_pcc:
            best_pcc = abs(current_pcc)
            best_perm = perm
    return  best_pcc, best_perm


def correlation_between_enhanced_images(X, transformed_RGB):
    X_gray = X['equa'].values
    best_pcc = 0
    best_perm = (0, 1, 2)
    for perm in permutations(range(3), 3):
        # reduce 3-dimension to 1 using UMAP again
        # transformer = umap.UMAP(n_neighbors=15, min_dist=0.2, n_components=1)
        spot_row = X['array_row'].values
        spot_col = X['array_col'].values

        max_row = np.int(np.max(spot_row))
        max_col = np.int(np.max(spot_col))

        img = np.ones(shape=(max_row + 1, max_col + 1, 3), dtype=np.float) * 255

        for index in range(len(transformed_RGB)):
            img[spot_row[index], spot_col[index]] = transformed_RGB[index]
        img = np.uint8(img)
        transformed_gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        transformed_gray_values = []
        for index in range(len(transformed_RGB)):
            transformed_gray_values.append(transformed_gray_img[spot_row[index], spot_col[index]])

        current_pcc, p_value = pearsonr(transformed_gray_values,  X_gray.tolist())
        if abs(current_pcc) > best_pcc:
            best_pcc = abs(current_pcc)
            best_perm = perm
        del img, transformed_gray_img, transformed_gray_values
    return best_pcc, best_perm


def nan_check_in_datframe():
    pass




'''
expression_file: file path for processed expression matrix
pca_conponent_num (50 in default): number of principle components remains in PCA denoising
umap_neighbor_num (10 in default): the size of the local neighborhood in UMAP
umap_min_dist (0.2 in default): control how tightly UMAP is allowed to pack points
plot_spot_radius (68 in default): spot radius in plotting pseudo-RGB image
'''
def transform_expression_to_RGB(expression_file, original_RGB = True, pca_conponent_num = 50,
                                pca_batch_size = 200, umap_neighbor_num = 10, umap_min_dist = 0.2,plot_spot_radius = 68):
    data = pd.read_csv(expression_file)

    X = data.loc[data['in_tissue'] != 0]

    # # read RGB columns to see if needing RGB optimization
    # if np.sum(X.loc[:, ['R','G','B']].values) != 0:
    #     original_RGB = True
    # else:
    #     original_RGB = False

    if 'equa' in data.columns:
        #set gene start idex
        gene_start_idx = 10
        #save enhanced image at spot level
        save_enhanced_RGB_to_image(X, expression_file)
    else:
        gene_start_idx = 9

    values = X.iloc[:, gene_start_idx:].values

    best_pcc = 0
    # best_pca_ratio = 0
    best_perm = (0,1,2)
    best_percentile_0 = 0
    best_percentile_1 = 0
    best_percentile_2 = 0

    # for pca_ratio in np.arange(0.75, 1.1, 0.01):
        # apply PCA to denoise first
    pac_model = IncrementalPCA(n_components=pca_conponent_num, batch_size= pca_batch_size)
    transformer = umap.UMAP(n_neighbors=umap_neighbor_num, min_dist=umap_min_dist, n_components=3)
    try:
        values = pac_model.fit_transform(val+ues)
        X_transformed = transformer.fit(values).transform(values)
    except Exception as e:
        print(str(e))
        return
    if original_RGB:
        # save original image at spot level
        save_spot_RGB_to_image(X, expression_file)
        #do grid search for the percentile parameters
        for percentile_0 in range(0, 50):
            for percentile_1 in range(0, 50):
                for percentile_2 in range(0, 50):
                    # plot_box_for_RGB(X_transformed)
                    # optimize pseudo-RGB according to spatial image
                    temp_X_transform = X_transformed.copy()
                    temp_X_transform[:, 0] = scale_to_RGB(X_transformed[:, 0], percentile_0)
                    temp_X_transform[:, 1] = scale_to_RGB(X_transformed[:, 1], percentile_1)
                    temp_X_transform[:, 2] = scale_to_RGB(X_transformed[:, 2], percentile_2)
                    if 'equa' in data.columns:
                        pcc, perm = correlation_between_enhanced_images(X, temp_X_transform)
                    else:

                        pcc, perm = RGB_correlation_between_images(X[['R','G','B']].values, temp_X_transform)


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

    save_transformed_RGB_to_image_and_csv(X,X_transformed[:, best_perm], expression_file, plot_spot_radius)

    del X_transformed, data, X, values, transformer





def plot_box_for_RGB(RGB_data):
    fig = plt.figure()
    plt.boxplot(RGB_data)
    plt.xticks([1,2,3], ['R\'','G\'','B\''],  rotation=0)
    plt.show()

def save_transformed_RGB_to_image_and_csv(X, X_transformed ,data_file_name, plot_spot_radius):

    # full size spot number
    spot_row = X['pxl_col_in_fullres'].values
    spot_col = X['pxl_row_in_fullres'].values

    max_row = np.int(np.max(spot_row + 1) + plot_spot_radius)
    max_col = np.int(np.max(spot_col + 1) + plot_spot_radius)

    img = np.ones(shape=(max_row + 1, max_col + 1, 3), dtype=np.uint8) * 255

    for index in range(len(X_transformed)):
        # img[spot_row[index], spot_col[index]] = values[index]
        # radius = 60
        # cv2.rectangle(img, (spot_col[index] - radius-7, spot_row[index] - radius),
        #               (spot_col[index] + radius+7, spot_row[index]+ radius),
        #               color=(int(values[index][0]), int(values[index][1]), int(values[index][2])),
        #               thickness=-1)
        cv2.circle(img, (spot_col[index], spot_row[index]), radius= plot_spot_radius,
                   color=(int(X_transformed[index][0]), int(X_transformed[index][1]), int(X_transformed[index][2])),
                   thickness=-1)



    cv2.imwrite(os.path.splitext(data_file_name)[0] + '_transformed.jpg', img)

    # save transformed_X back to csv file
    # X.insert(gene_start_idx, 'transformed_R',X_transformed[:, 0], True)
    # X.insert(gene_start_idx+1, 'transformed_G',X_transformed[:, 1], True)
    # X.insert(gene_start_idx+2, 'transformed_B',X_transformed[:, 2], True)
    #
    # # #then save to csv file
    # X.iloc[:, 0:gene_start_idx+3].to_csv(os.path.splitext(data_file_name)[0] + '_newRGB.csv', index = False)

    del img, spot_row, spot_col#, mask, inpaint_img,  resize_contour#resize_img,

def save_pseudoRGB_to_spot_level(X, gene_start_idx, data_path):
    values = X.iloc[:, gene_start_idx:].values
    spot_row  = X['pxl_col_in_fullres'].values
    spot_col = X['pxl_row_in_fullres'].values
    max_row = np.int(np.max(spot_row + 1))
    max_col = np.int(np.max(spot_col + 1))
    img = np.ones(shape=(max_row + 1, max_col + 1, 3), dtype=np.uint8) * 255
    for index in range(len(values)):
        cv2.circle(img, (spot_col[index], spot_row[index]), radius= 68,
                   color=(int(values[index][0]), int(values[index][1]), int(values[index][2])),
                   thickness=-1)
    cv2.imwrite(os.path.splitext(data_path)[0] + '_spot_circle.jpg', img)
    del values, spot_row, max_col, img

def save_spot_RGB_to_image(X, data_file ):

    values = X[['R','G','B']].values
    spot_row = X['array_row'].values
    spot_col = X['array_col'].values

    max_row = np.int(np.max(spot_row))
    max_col = np.int(np.max(spot_col))

    img = np.ones(shape=(max_row+1, max_col+1,  3), dtype= np.float) * 255

    for index in range(len(values)):
        img[spot_row[index], spot_col[index]] = values[index]

    resize_img = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.splitext(data_file)[0] + '_raw.jpg', resize_img)
    del values, spot_row, max_col, img, resize_img

def save_enhanced_RGB_to_image(X, data_file):

    values = X['equa'].values
    spot_row = X['array_row'].values
    spot_col = X['array_col'].values

    max_row = np.int(np.max(spot_row))
    max_col = np.int(np.max(spot_col))

    img = np.ones(shape=(max_row+1, max_col+1,  1), dtype= np.float) * 255

    for index in range(len(values)):
        img[spot_row[index], spot_col[index]] = values[index]

    resize_img = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.splitext(data_file)[0] + '_enhanced.jpg', resize_img)
    del spot_row, max_col, img, resize_img

if __name__ == '__main__':
    #
    # data_folder = r'/home/fei/Desktop/test/Brain/'
    # for expression_file in os.listdir(data_folder):
    #     if os.path.splitext(expression_file)[1] == '.csv':
    #         transform_expression_to_RGB(os.path.join(data_folder, expression_file))

    data_file_name = r'/home/fei/Desktop/single_cell_spatial_image-main/vel_data.csv'
    transform_expression_to_RGB(data_file_name, True, pca_conponent_num=10)
