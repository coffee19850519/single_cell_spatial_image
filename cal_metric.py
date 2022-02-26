import os.path as osp
import pickle
import shutil
import tempfile
import os
import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
import pandas as pd
import json
import cv2
from PIL import Image
from sklearn.metrics.cluster import adjusted_rand_score
import shutil
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score, fowlkes_mallows_score, rand_score, \
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
import math
from mmseg.apis.inference import inference_segmentor
from mmseg.apis.inference import init_segmentor

def testing_metric(adata,img_path, output_folder, model, show_dir, k):
    MI_list = [] 
    name_list = []
    k_list = []
    if k==-1:
        for name in os.listdir(img_path):
            MI_max = 0  
            img_name = img_path+name
            result = inference_segmentor(model, img_name, k)
            out_file=show_dir+name
            image_test = cv2.imread(img_name)
            MI = cluster_heterogeneity(image_test, result[0], 0)
            if MI_max < MI:
                MI_max = MI
                optimal_name = name 

        for tmp_k in range(4, 10):
            MI_max = 0     
            img_name = img_path+optimal_name       
            result = inference_segmentor(model, img_name, tmp_k)
            out_file=show_dir+name
            model.show_result(
                                    img_name,
                                    result,
                                    palette=None,
                                    show=False,
                                    out_file=out_file)

            image_test = cv2.imread(img_name)
            if not os.path.exists(output_folder+'result_temp/'):
                os.makedirs(output_folder+'result_temp/')
            np.savetxt(output_folder+'result_temp/'+name.split('.png')[0]+'.csv', result[0], delimiter=',')
            MI = cluster_heterogeneity(image_test, result[0], 0)
            if MI_max < MI:
                MI_max = MI
                optimal_k = tmp_k

        result = inference_segmentor(model, img_name, optimal_k)
        model.show_result(
                                img_name,
                                result,
                                palette=None,
                                show=False,
                                out_file=out_file)
        k_list.append(optimal_k)
        name_list.append(optimal_name)
        MI_list.append(MI_max)


        MI_result = {
            'name': name_list,
            'k':k_list,
            'MI': MI_list,
        }
        MI_result = pd.DataFrame(MI_result)
        MI_result = MI_result.sort_values(by=['MI'], ascending=False)

        if len(name_list) > 5:
            MI_result_top5 = MI_result[0:5]
            name = MI_result_top5.iloc[:, 0].values
            for n in name:
                prefix = n.split('.png')[0]
                show = cv2.imread(show_dir + n)
                if not os.path.exists(output_folder + 'segmentation_map/'):
                    os.makedirs(output_folder + 'segmentation_map/')
                cv2.imwrite(output_folder + 'segmentation_map/' + n, show)

                if not os.path.exists(output_folder+'result/'):
                    os.makedirs(output_folder+'result/')
                shutil.move(output_folder+'result_temp/'+prefix+'.csv', output_folder+'result/'+prefix+'.csv')
                category_map = pd.read_csv(output_folder+'result/'+prefix+'.csv',header =None)
                get_spot_category(adata, category_map, 'vote',prefix)

            shutil.rmtree(show_dir)
            shutil.rmtree(output_folder+'result_temp/')
            adata.obs.to_csv(output_folder + 'predicted_tissue_architecture.csv')
            MI_result_top5.to_csv(output_folder + 'top5_MI_value.csv', index=True, header=True)
        else:
            name = MI_result.iloc[:, 0].values
            for n in name:
                prefix = n.split('.png')[0]
                show = cv2.imread(show_dir + n)
                if not os.path.exists(output_folder + 'segmentation_map/'):
                    os.makedirs(output_folder + 'segmentation_map/')
                cv2.imwrite(output_folder + 'segmentation_map/' + n, show)

                if not os.path.exists(output_folder + 'result/'):
                    os.makedirs(output_folder + 'result/')
                shutil.move(output_folder + 'result_temp/' + prefix + '.csv', output_folder + 'result/' + prefix + '.csv')
                category_map = pd.read_csv(output_folder+'result/'+prefix+'.csv',header =None)
                get_spot_category(adata, category_map, 'vote',prefix)

            shutil.rmtree(show_dir)
            shutil.rmtree(output_folder + 'result_temp/')
            adata.obs.to_csv(output_folder + 'predicted_tissue_architecture.csv')
            MI_result.to_csv(output_folder + 'top5_MI_value.csv', index=True, header=True)

        top1_name = MI_result.iloc[:, 0].values[0]
        top1_csv_name = output_folder + 'result/' + top1_name.split('.png')[0] + '.csv'
        top1_category_map = np.loadtxt(top1_csv_name,dtype=np.int32, delimiter=",")
    else:
        for name in os.listdir(img_path):
            img_name = img_path+name
            name_list.append(name)
            result = inference_segmentor(model, img_name, k)

            out_file=show_dir+name
            print(out_file)
            model.show_result(
                                    img_name,
                                    result,
                                    palette=None,
                                    show=False,
                                    out_file=out_file)

            image_test = cv2.imread(img_name)
            if not os.path.exists(output_folder+'result_temp/'):
                os.makedirs(output_folder+'result_temp/')
            np.savetxt(output_folder+'result_temp/'+name.split('.png')[0]+'.csv', result[0], delimiter=',')

            MI = cluster_heterogeneity(image_test, result[0], 0)
            MI_list.append(MI)

        MI_result = {
            'name': name_list,
            'MI': MI_list,
        }
        MI_result = pd.DataFrame(MI_result)
        MI_result = MI_result.sort_values(by=['MI'], ascending=False)

        if len(name_list) > 5:
            MI_result_top5 = MI_result[0:5]
            # print(MI_result_top5)
            name = MI_result_top5.iloc[:, 0].values
            for n in name:
                prefix = n.split('.png')[0]
                show = cv2.imread(show_dir + n)
                if not os.path.exists(output_folder + 'segmentation_map/'):
                    os.makedirs(output_folder + 'segmentation_map/')
                cv2.imwrite(output_folder + 'segmentation_map/' + n, show)

                if not os.path.exists(output_folder+'result/'):
                    os.makedirs(output_folder+'result/')
                shutil.move(output_folder+'result_temp/'+prefix+'.csv', output_folder+'result/'+prefix+'.csv')
                category_map = pd.read_csv(output_folder+'result/'+prefix+'.csv',header =None)
                get_spot_category(adata, category_map, 'vote',prefix)

            shutil.rmtree(show_dir)
            shutil.rmtree(output_folder+'result_temp/')
            adata.obs.to_csv(output_folder + 'predicted_tissue_architecture.csv')
            MI_result_top5.to_csv(output_folder + 'top5_MI_value.csv', index=True, header=True)
        else:
            name = MI_result.iloc[:, 0].values
            for n in name:
                prefix = n.split('.png')[0]
                show = cv2.imread(show_dir + n)
                if not os.path.exists(output_folder + 'segmentation_map/'):
                    os.makedirs(output_folder + 'segmentation_map/')
                cv2.imwrite(output_folder + 'segmentation_map/' + n, show)

                if not os.path.exists(output_folder + 'result/'):
                    os.makedirs(output_folder + 'result/')
                shutil.move(output_folder + 'result_temp/' + prefix + '.csv', output_folder + 'result/' + prefix + '.csv')
                category_map = pd.read_csv(output_folder+'result/'+prefix+'.csv',header =None)
                get_spot_category(adata, category_map, 'vote',prefix)

            shutil.rmtree(show_dir)
            shutil.rmtree(output_folder + 'result_temp/')
            adata.obs.to_csv(output_folder + 'predicted_tissue_architecture.csv')
            MI_result.to_csv(output_folder + 'top5_MI_value.csv', index=True, header=True)

        top1_name = MI_result.iloc[:, 0].values[0]
        top1_csv_name = output_folder + 'result/' + top1_name.split('.png')[0] + '.csv'
        top1_category_map = np.loadtxt(top1_csv_name,dtype=np.int32, delimiter=",")

    shutil.rmtree(output_folder + 'result/')
    return top1_category_map

def evaluation_metric(adata, img_path, output_folder, model, show_dir, label_path, k):
    MI_list = []
    name_list = []
    ARI_list = []
    AMI_list = []
    FMI_list = []
    RI_list = []
    k_list = []
    if k == -1:
        for name in os.listdir(img_path):
            MI_max = 0
            img_name = img_path+name
            result = inference_segmentor(model, img_name, k)
            out_file=show_dir+name
            image_test = cv2.imread(img_name)
            MI = cluster_heterogeneity(image_test, result[0], 0)
            if MI_max < MI:
                MI_max = MI
                optimal_name = name 


        for tmp_k in range(4, 10):
            MI_max = 0     
            img_name = img_path+optimal_name       
            result = inference_segmentor(model, img_name, tmp_k)
            out_file=show_dir+name
            model.show_result(
                                    img_name,
                                    result,
                                    palette=None,
                                    show=False,
                                    out_file=out_file)

            image_test = cv2.imread(img_name)
            if not os.path.exists(output_folder+'result_temp/'):
                os.makedirs(output_folder+'result_temp/')
            np.savetxt(output_folder+'result_temp/'+name.split('.png')[0]+'.csv', result[0], delimiter=',')
            MI = cluster_heterogeneity(image_test, result[0], 0)
            name0, ARI, AMI, FMI, RI = calculate(adata, result[0], img_name, label_path)

            if MI_max < MI:

                MI_max = MI
                optimal_k = tmp_k
                optimal_ARI = ARI
                optimal_MI = MI
                optimal_AMI = AMI
                optimal_FMI = FMI
                optimal_RI = RI

        result = inference_segmentor(model, img_name, optimal_k)
        model.show_result(
                                img_name,
                                result,
                                palette=None,
                                show=False,
                                out_file=out_file)
        k_list.append(optimal_k)
        name_list.append(optimal_name)
        MI_list.append(optimal_MI)
        ARI_list.append(optimal_ARI)
        AMI_list.append(optimal_AMI)
        FMI_list.append(optimal_FMI)
        RI_list.append(optimal_RI)

        MI_result = {
                    'name': name_list,
                    'k':k_list,
                    "ARI": ARI_list,
                    "AMI": AMI_list,
                    "FMI": FMI_list,
                    "RI": RI_list,
                    'MI': MI_list,

                }
        MI_result = pd.DataFrame(MI_result)
        MI_result = MI_result.sort_values(by=['MI'], ascending=False)
        
        if len(name_list) > 5:
            MI_result_top5 = MI_result[0:5]
            # print(MI_result_top5)
            name = MI_result_top5.iloc[:, 0].values
            for n in name:
                prefix = n.split('.png')[0]
                show = cv2.imread(show_dir + n)
                if not os.path.exists(output_folder + 'segmentation_map/'):
                    os.makedirs(output_folder + 'segmentation_map/')
                cv2.imwrite(output_folder + 'segmentation_map/' + n, show)

                if not os.path.exists(output_folder+'result/'):
                    os.makedirs(output_folder+'result/')
                shutil.move(output_folder+'result_temp/'+prefix+'.csv', output_folder+'result/'+prefix+'.csv')
                category_map = pd.read_csv(output_folder+'result/'+prefix+'.csv',header =None)
                get_spot_category(adata, category_map, 'vote',prefix)

            shutil.rmtree(show_dir)
            shutil.rmtree(output_folder+'result_temp/')
            adata.obs.to_csv(output_folder + 'predicted_tissue_architecture.csv')
            MI_result_top5.to_csv(output_folder + 'top5_evaluation.csv', index=True, header=True)
        else:
            name = MI_result.iloc[:, 0].values
            for n in name:
                prefix = n.split('.png')[0]
                show = cv2.imread(show_dir + n)
                if not os.path.exists(output_folder + 'segmentation_map/'):
                    os.makedirs(output_folder + 'segmentation_map/')
                cv2.imwrite(output_folder + 'segmentation_map/' + n, show)

                if not os.path.exists(output_folder + 'result/'):
                    os.makedirs(output_folder + 'result/')
                shutil.move(output_folder + 'result_temp/' + prefix + '.csv', output_folder + 'result/' + prefix + '.csv')
                category_map = pd.read_csv(output_folder+'result/'+prefix+'.csv',header =None)
                get_spot_category(adata, category_map, 'vote',prefix)

            shutil.rmtree(show_dir)
            shutil.rmtree(output_folder + 'result_temp/')
            adata.obs.to_csv(output_folder + 'predicted_tissue_architecture.csv')
            MI_result.to_csv(output_folder + 'top5_evaluation.csv', index=True, header=True)

        top1_name = MI_result.iloc[:, 0].values[0]
        top1_csv_name = output_folder + 'result/' + top1_name.split('.png')[0] + '.csv'
        top1_category_map = np.loadtxt(top1_csv_name,dtype=np.int32, delimiter=",")
        # shutil.rmtree(output_folder + 'result/')
    else:
        for name in os.listdir(img_path):
            img_name = img_path+name
            name_list.append(name)

            result = inference_segmentor(model, img_name, k)
            name0, ARI, AMI, FMI, RI = calculate(adata, result[0], img_name, label_path)
            ARI_list.append(ARI)
            AMI_list.append(AMI)
            FMI_list.append(FMI)
            RI_list.append(RI)
            # print(result[0])
            print(img_name)
            out_file=show_dir+name
            # print(out_file)
            model.show_result(
                                    img_name,
                                    result,
                                    palette=None,
                                    show=False,
                                    out_file=out_file)

            image_test = cv2.imread(img_name)
            if not os.path.exists(output_folder+'result_temp/'):
                os.makedirs(output_folder+'result_temp/')
            np.savetxt(output_folder+'result_temp/'+name.split('.png')[0]+'.csv', result[0], delimiter=',')

            MI = cluster_heterogeneity(image_test, result[0], 0)
            MI_list.append(MI)

        MI_result = {
                    'name': name_list,
                    "ARI": ARI_list,
                    "AMI": AMI_list,
                    "FMI": FMI_list,
                    "RI": RI_list,
                    'MI': MI_list,

                }
        MI_result = pd.DataFrame(MI_result)
        MI_result = MI_result.sort_values(by=['MI'], ascending=False)
        
        if len(name_list) > 5:
            MI_result_top5 = MI_result[0:5]
            # print(MI_result_top5)
            name = MI_result_top5.iloc[:, 0].values
            for n in name:
                prefix = n.split('.png')[0]
                show = cv2.imread(show_dir + n)
                if not os.path.exists(output_folder + 'segmentation_map/'):
                    os.makedirs(output_folder + 'segmentation_map/')
                cv2.imwrite(output_folder + 'segmentation_map/' + n, show)

                if not os.path.exists(output_folder+'result/'):
                    os.makedirs(output_folder+'result/')
                shutil.move(output_folder+'result_temp/'+prefix+'.csv', output_folder+'result/'+prefix+'.csv')
                category_map = pd.read_csv(output_folder+'result/'+prefix+'.csv',header =None)
                get_spot_category(adata, category_map, 'vote',prefix)

            shutil.rmtree(show_dir)
            shutil.rmtree(output_folder+'result_temp/')
            adata.obs.to_csv(output_folder + 'predicted_tissue_architecture.csv')
            MI_result_top5.to_csv(output_folder + 'top5_evaluation.csv', index=True, header=True)
        else:
            name = MI_result.iloc[:, 0].values
            for n in name:
                prefix = n.split('.png')[0]
                show = cv2.imread(show_dir + n)
                if not os.path.exists(output_folder + 'segmentation_map/'):
                    os.makedirs(output_folder + 'segmentation_map/')
                cv2.imwrite(output_folder + 'segmentation_map/' + n, show)

                if not os.path.exists(output_folder + 'result/'):
                    os.makedirs(output_folder + 'result/')
                shutil.move(output_folder + 'result_temp/' + prefix + '.csv', output_folder + 'result/' + prefix + '.csv')
                category_map = pd.read_csv(output_folder+'result/'+prefix+'.csv',header =None)
                get_spot_category(adata, category_map, 'vote',prefix)

            shutil.rmtree(show_dir)
            shutil.rmtree(output_folder + 'result_temp/')
            adata.obs.to_csv(output_folder + 'predicted_tissue_architecture.csv')
            MI_result.to_csv(output_folder + 'top5_evaluation.csv', index=True, header=True)

        top1_name = MI_result.iloc[:, 0].values[0]
        top1_csv_name = output_folder + 'result/' + top1_name.split('.png')[0] + '.csv'
        top1_category_map = np.loadtxt(top1_csv_name,dtype=np.int32, delimiter=",")
    shutil.rmtree(output_folder + 'result/')
    return top1_category_map


def cluster_heterogeneity(image_test, category_map, background_category):
    if len(category_map.shape) > 2:
        category_map = cv2.cvtColor(category_map, cv2.COLOR_BGR2GRAY)
    category_list = np.unique(category_map)

    W = np.zeros((len(category_list), len(category_list)))
    for i in range(category_map.shape[0]):
        flag1 = category_map[i][0]
        flag2 = category_map[0][i]
        for j in range(category_map.shape[0]):
            if category_map[i][j] != flag1:  # for row
                index1 = np.where(category_list == flag1)[0][0]
                index2 = np.where(category_list == category_map[i][j])[0][0]
                W[index1][index2] = 1
                W[index2][index1] = 1
                flag1 = category_map[i][j]
            if category_map[j][i] != flag2:  # for column
                index1 = np.where(category_list == flag2)[0][0]
                index2 = np.where(category_list == category_map[j][i])[0][0]
                W[index1][index2] = 1
                W[index2][index1] = 1
                flag2 = category_map[j][i]
    W = W[1:, 1:]  #
    # print(W)
    category_num = W.shape[0]


    # print(R.shape)
    MI_list = []
    image_test_ori = image_test
    # Calculate the average color value of each channel in each cluster
    for channel in range(3):
        image_test = image_test_ori[:, :, channel]
        # print(image_test)
        num = 0
        gray_list = []
        gray_mean = 0
        for category in category_list:
            pixel_x, pixel_y = np.where(category_map == category)
            if category == background_category:
                num = len(pixel_x)
                continue
            gray = []
            for i in range(len(pixel_x)):
                gray.append(image_test[pixel_x[i], pixel_y[i]])
            gray_value = np.mean(gray)
            gray_list.append(gray_value)
            gray_mean += gray_value * len(pixel_x)
        gray_mean = gray_mean / (image_test.shape[0] ** 2 - num)

        n = W.shape[0]
        a = 0
        b = 0
        for p in range(n):
            index, = np.where(W[p] == 1)
            for q in range(len(index)):
                a += abs((gray_list[p] - gray_mean) * (gray_list[index[q]] - gray_mean))
            b += (gray_list[p] - gray_mean) ** 2
        MI = n * a / (b * np.sum(W))
        MI_list.append(MI)
    # print(MI_list)
    MI = math.sqrt((MI_list[0] ** 2 + MI_list[1] ** 2 + MI_list[2] ** 2) / 3)
    # print(MI)
    return MI



def calculate(adata, output, img_path, label_path):
    img_name = img_path.split('/')[-1]  # eg:151507_50_32_....png

    samples_num = img_name.split('_')[0]  # eg:151507

    labels = save_spot_RGB_to_image(label_path, adata)  # label
    label = labels.flatten().tolist()
    output = np.array(output).flatten().tolist()
    # print('len(output)',len(output))

    label_final = []
    output_final = []
    shape = adata.uns["img_shape"]
    for i in range(shape ** 2):
        if label[i] != 0:
            label_final.append(label[i])
            output_final.append(output[i])
    ARI = adjusted_rand_score(label_final, output_final)
    AMI = adjusted_mutual_info_score(label_final, output_final)
    FMI = fowlkes_mallows_score(label_final, output_final)
    RI = rand_score(label_final, output_final)
    print('name', img_name)
    print('ARI:', ARI)

    return img_name, ARI, AMI, FMI, RI


def save_spot_RGB_to_image(label_path, adata):
    # data_file = os.path.join(data_folder, expression_file)
    X = pd.read_csv(label_path)
    X = X.sort_values(by=['barcode'])
    assert all(adata.obs.index == X.iloc[:, 0].values)
    layers = X.iloc[:, 1].values
    # print(layers)
    spot_row = adata.obs["pxl_col_in_fullres"]
    spot_col = adata.obs["pxl_row_in_fullres"]

    radius = int(0.5 * adata.uns['fiducial_diameter_fullres'] + 1)
    # radius = int(scaler['spot_diameter_fullres'] + 1)
    max_row = max_col = int((2000 / adata.uns['tissue_hires_scalef']) + 1)
    # radius = round(radius * (600 / 2000))
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

def get_spot_category_by_center_pixel(category_map, center_x, center_y):

    return category_map[center_x, center_y]

def get_spot_category_by_pixel_vote(category_map, center_x, center_y,max_row, max_col, radius):

    spot_region_start_x = center_x - radius
    spot_region_end_x = center_x + radius
    spot_region_start_y = center_y - radius
    spot_region_end_y = center_y + radius

    if spot_region_start_x < 0:
        spot_region_start_x = 0
    if spot_region_start_y < 0:
        spot_region_start_y = 0
    if spot_region_end_x > max_row:
        spot_region_end_x = max_row
    if spot_region_end_y > max_col:
        spot_region_end_y = max_col

    spot_region = category_map.values[spot_region_start_x:  spot_region_end_x,spot_region_start_y: spot_region_end_y]
    # print(spot_region)
    categories, votes = np.unique(spot_region, return_counts=True)

    return int(categories[np.argmax(votes)])

def get_spot_category(adata, category_map, strategy,name):
    predict = []
    #infer resolution
    if category_map.shape[0] == 600:
        # low resolution
        resolution = 'low'
        radius = int((0.5 * adata.uns['fiducial_diameter_fullres'] + 1) * adata.uns['tissue_lowres_scalef'])
        max_row = max_col = 600
    elif category_map.shape[0] == 400:
        # low resolution
        resolution = 'low'
        radius = int((0.5 * adata.uns['fiducial_diameter_fullres'] + 1) * adata.uns['tissue_lowres_scalef'])
        max_row = 400
        max_col = 600
    elif category_map.shape[0] == 2000:
        #high resolution
        resolution = 'high'
        radius = int((0.5 * adata.uns['fiducial_diameter_fullres'] + 1) * adata.uns['tissue_hires_scalef'])
        max_row = max_col = 2000
    else:
        #full resolution
        resolution = 'full'
        radius = int(0.5 * adata.uns['fiducial_diameter_fullres'] + 1)
        max_row = max_col = int((2000 / adata.uns['tissue_hires_scalef']) + 1)
    for index, row in adata.obs.iterrows():
        if resolution == 'low':
            center_x = int((row['pxl_col_in_fullres'] /(2000 / adata.uns['tissue_hires_scalef'] + 1)) *600)
            center_y = int((row['pxl_row_in_fullres'] /(2000 / adata.uns['tissue_hires_scalef'] + 1)) *600)
            # print(center_x, center_y)
        elif resolution == 'high':
            center_x = int((row['pxl_col_in_fullres'] /(2000 / adata.uns['tissue_hires_scalef'] + 1)) *2000)
            center_y = int((row['pxl_row_in_fullres'] /(2000 / adata.uns['tissue_hires_scalef'] + 1)) *2000)
        else:
            center_x = row['pxl_col_in_fullres']
            center_y = row['pxl_row_in_fullres']


        if strategy == 'vote':
            predictive_layer = get_spot_category_by_pixel_vote(category_map,
                                                                      center_x, center_y, max_row,
                                                                      max_col,radius)
            # row[col_name] = predictive_layer
            predict.append(predictive_layer)
        else:
            predictive_layer = get_spot_category_by_center_pixel(category_map,
                                                                        center_x, center_y)
            predict.append(predictive_layer)
    col_name = 'predicted_category_'+name
    adata.obs[col_name] = predict
