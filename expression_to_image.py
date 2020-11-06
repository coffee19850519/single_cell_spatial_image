import pandas as pd
import umap
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from itertools import permutations

def scale_to_RGB(channel,truncated_percent):
    truncated_down = np.percentile(channel, truncated_percent)
    truncated_up = np.percentile(channel, 100 - truncated_percent)
    channel_new = (255 / (truncated_up - truncated_down)) * channel
    channel_new[channel_new < 0] = 0
    channel_new[channel_new > 255] = 255
    return np.uint8(channel_new)

def correlation_between_images(X_RGB, transformed_RGB):
    best_pcc = 0
    for perm in permutations(range(3), 3):
        current_pcc, p_value = pearsonr(X_RGB.reshape((-1)), transformed_RGB[:, perm].reshape((-1)))
        if abs(current_pcc) > best_pcc:
            best_pcc =  abs(current_pcc)

    return  best_pcc



def transform_expression_to_RGB(expression_file):
    data = pd.read_csv(expression_file, sep = '\t')

    X = data.loc[data['in_tissue'] != 0]
    values = X.iloc[:, 10:].values
    transformer = umap.UMAP(n_neighbors= 10, min_dist= 0.2, n_components=3, metric='cosine')

    X_transformed = transformer.fit(values).transform(values)

    # plot_box_for_RGB(X_transformed)
    X_transformed[:, 0] = scale_to_RGB(X_transformed[:, 0],5)
    X_transformed[:, 1] = scale_to_RGB(X_transformed[:, 1],5)
    X_transformed[:, 2] = scale_to_RGB(X_transformed[:, 2],5)

    corr = correlation_between_images(X.iloc[:, 7:10].values, X_transformed)

    print(corr)

    save_transformed_RGB_to_image(X, X_transformed, expression_file)

    del X_transformed, data, X, values, transformer

def plot_box_for_RGB(RGB_data):
    fig = plt.figure()
    plt.boxplot(RGB_data)
    plt.xticks([1,2,3], ['R\'','G\'','B\''],  rotation=0)
    plt.show()

def save_transformed_RGB_to_image(X, transformed_X, data_file_name):
    spot_row = X.iloc[:, 3].values
    spot_col = X.iloc[:, 4].values

    max_row = np.max(spot_row)
    max_col = np.max(spot_col)

    img = np.ones(shape=(max_row + 1, max_col + 1, 3), dtype=np.float) * 255
    # should follow the spot permutation at full resolution image
    for index in range(len(transformed_X)):
        img[spot_row[index], spot_col[index]] = transformed_X[index]
    resize_img = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.splitext(data_file_name)[0] + '.jpg', resize_img)
    del img, spot_row, spot_col, resize_img

def save_spot_RGB_to_image(data_file):
    data = pd.read_csv(data_file, sep='\t')

    X = data.loc[data['in_tissue'] != 0]
    values = X.iloc[:,7:10].values
    spot_row = X.iloc[:, 3].values
    spot_col = X.iloc[:, 4].values

    max_row = np.max(spot_row)
    max_col = np.max(spot_col)

    img = np.ones(shape=(max_row+1, max_col+1,  3), dtype= np.float) * 255
    for index in range(len(values)):
        img[spot_row[index], spot_col[index]] = values[index]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.splitext(data_file)[0] + '.jpg', img)
    del data, X, values, spot_row, max_col, img

data_folder = r'/home/fei/Desktop/data/'
for expression_file in os.listdir(data_folder):
    if os.path.splitext(expression_file)[1] == '.csv':
        transform_expression_to_RGB(os.path.join(data_folder, expression_file))
        # save_spot_RGB_to_image(os.path.join(data_folder, expression_file))