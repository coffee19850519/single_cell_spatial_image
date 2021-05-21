import pandas as pd
import scipy.sparse
import numpy as np
from sklearn.decomposition import IncrementalPCA
import umap, os, json
import cv2
from scipy.stats import pearsonr
from itertools import permutations

def scale_to_RGB(channel,truncated_percent):
    truncated_down = np.percentile(channel, truncated_percent)
    truncated_up = np.percentile(channel, 100 - truncated_percent)
    channel_new = ((channel - truncated_down) / (truncated_up - truncated_down)) * 255
    channel_new[channel_new < 0] = 0
    channel_new[channel_new > 255] = 255
    return np.uint8(channel_new)


def save_transformed_RGB_to_image_and_csv(spot_row_in_fullres,
                                          spot_col_in_fullres,
                                          max_row, max_col,
                                          X_transformed,
                                          sample_name,
                                          img_type,
                                          img_folder,
                                          plot_spot_radius,
                                          ):
  
    img = np.ones(shape=(max_row + 1, max_col + 1, 3), dtype=np.uint8) * 255

    for index in range(len(X_transformed)):

        cv2.rectangle(img, (spot_col_in_fullres[index] - plot_spot_radius, spot_row_in_fullres[index] - plot_spot_radius),
                      (spot_col_in_fullres[index] + plot_spot_radius, spot_row_in_fullres[index]+ plot_spot_radius),
                      color=(int(X_transformed[index][0]), int(X_transformed[index][1]), int(X_transformed[index][2])),
                      thickness=-1)
    #optional  both/high/low/none
    hi_img = cv2.resize(img, dsize=(2000, 2000), interpolation=cv2.INTER_CUBIC)
    low_img = cv2.resize(img, dsize=(600, 600), interpolation=cv2.INTER_CUBIC)
    # sample_num = sample_name.split('_')[0]
    image_path = img_folder+'/pseudo_images/'
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if img_type == 'lowres':
        cv2.imwrite(image_path+sample_name + '_transformed_lowres.png', low_img)
        # img_path = 'pseudo_image/'+sample_name + '_transformed_lowres.jpg'
        # img_path = 'pseudo_image'
    elif img_type == 'hires':
        cv2.imwrite(image_path+sample_name + '_transformed_hires.jpg', hi_img)
        # img_path = 'pseudo_image/'+sample_name + '_transformed_hires.jpg'
    elif img_type == 'both':
        cv2.imwrite(image_path+sample_name + '_transformed_hires.jpg', hi_img)
        cv2.imwrite(image_path+sample_name + '_transformed_lowres.jpg', low_img)
        # img_path = 'pseudo_image/'+sample_name + '_transformed_lowres.jpg'

    # cv2.imwrite(sample_name + '_transformed_hires.jpg', hi_img)
    # cv2.imwrite(sample_name + '_transformed_lowres.jpg', low_img)
    # img_path = sample_name + '_transformed_lowres.jpg'
    del img, spot_row_in_fullres, spot_col_in_fullres
    return hi_img, low_img

