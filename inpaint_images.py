import os
import cv2
import numpy as np
import pandas as pd
import json
from PIL import Image
h = 600



def save_spot_RGB_to_image(anndata,metadata_all,img_col,img_row):

    X = metadata_all
    # print(X)
    tissue = X.iloc[:, 0].values

    radius = int(0.5 * anndata.uns['fiducial_diameter_fullres'] + 1)
    max_row = max_col = int((2000 / anndata.uns['tissue_hires_scalef']) + 1)
    radius = round(radius * (img_row/max_row))
    spot_row = X.iloc[:, 3].values * (img_row/max_row)
    spot_row = spot_row.astype(np.int)
    spot_col = X.iloc[:, 4].values * (img_col/max_col)
    spot_col = spot_col.astype(np.int)

    in_tissue = []
    out_tissue = []
    for index in range(len(spot_row)):
        if tissue[index] == 1:
            in_tissue.append((spot_row[index], spot_col[index]))
        if tissue[index] == 0:
            out_tissue.append((spot_row[index], spot_col[index]))


    return in_tissue, out_tissue, radius


def KNN(point,point_list,num):
    point = np.array(point)
    point_list = np.array(point_list)
    distance = np.sqrt(np.sum(np.asarray(point - point_list) ** 2, axis=1))
    index_sort = np.argsort(distance)
    index = index_sort[:num]
    near_spot = []
    weight = []
    for i in range(num):
        near_spot.append(tuple(point_list[index[i]]))
        weight.append(1/(distance[index[i]]))
    exp_weight = np.exp(weight)
    weight = (exp_weight / np.sum(exp_weight)).tolist()
    return near_spot,weight


def inpaint(img_path, sample, anndata, metadata_all):
    for name in os.listdir(img_path):
        if name.split('_',5)[0]==sample:
            img = cv2.imread(os.path.join(img_path, name))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, binary = cv2.threshold(gray,230,255,cv2.THRESH_BINARY_INV)
            contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            area = []
            for k in range(len(contours)):
                area.append(cv2.contourArea(contours[k]))
            max_idx = np.argmax(np.array(area))
            img_row = img.shape[0]
            img_col = img.shape[1]

            in_tissue, out_tissue, radius = save_spot_RGB_to_image(anndata,metadata_all,img_col,img_row)
            radius += 2
            sample = name.split('_')[0]
            k = 6
            inpaint_path = img_path
            for i in range(len(out_tissue)):
                dist = cv2.pointPolygonTest(contours[max_idx], (int(out_tissue[i][1]),int(out_tissue[i][0])),False)
                if dist == 1.0:
                    nn_list, weight = KNN(out_tissue[i], in_tissue, k)
                    pixel_sum = []
                    for j in range(k):
                        pixel_sum.append(img[nn_list[j][0]][nn_list[j][1]]*weight[j])
                    pixel_sum_r = np.array(pixel_sum)[:, 0].sum()
                    pixel_sum_g = np.array(pixel_sum)[:, 1].sum()
                    pixel_sum_b = np.array(pixel_sum)[:, 2].sum()
                    pixel = (pixel_sum_r, pixel_sum_g, pixel_sum_b)
                    img[(out_tissue[i][0] - radius):(out_tissue[i][0] + radius),(out_tissue[i][1] - radius):(out_tissue[i][1] + radius)] = pixel
                    if not os.path.exists(inpaint_path):
                        os.makedirs(inpaint_path)
                    cv2.imwrite(inpaint_path +name, img)

    return inpaint_path
