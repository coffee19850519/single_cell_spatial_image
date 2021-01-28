import collections
import scipy.sparse as sp_sparse
import tables
import numpy as np
import pandas as pd
import os
import glob
import cv2
import json

print(os.path.abspath(os.getcwd()))
print("suppose to be '/group/xulab/Su_Li/Yuzhou_sptl/RNA_Velocity")

List_Velocity = ["1.0.0/V1_Adult_Mouse_Brain",
                 "1.0.0/V1_Mouse_Brain_Sagittal_Anterior",
                 "1.0.0/V1_Mouse_Brain_Sagittal_Anterior_Section_2",
                 "1.0.0/V1_Mouse_Brain_Sagittal_Posterior",
                 "1.0.0/V1_Mouse_Brain_Sagittal_Posterior_Section_2",
                 "1.0.0/V1_Mouse_Kidney",
                 "1.1.0/V1_Adult_Mouse_Brain",
                 #"1.1.0/V1_Adult_Mouse_Brain_Coronal_Section_1",
                 #"1.1.0/V1_Adult_Mouse_Brain_Coronal_Section_2",
                 "1.1.0/V1_Mouse_Brain_Sagittal_Anterior",
                 "1.1.0/V1_Mouse_Brain_Sagittal_Anterior_Section_2",
                 "1.1.0/V1_Mouse_Brain_Sagittal_Posterior",
                 "1.1.0/V1_Mouse_Brain_Sagittal_Posterior_Section_2",
                 "1.1.0/V1_Mouse_Kidney"]
for item in List_Velocity:

    velocity_input = glob.glob("/group/xulab/wangjue/spatialData/RNA_Velocity/" + str(item.split('/')[1]) + "_possorted_genome_bam1*")[0]

    spatial_csv = glob.glob("/group/xulab/Su_Li/Yuzhou_sptl/Data_source/Spatial_Gene_expression1.0.0/Mouse Brain Section (Coronal)" + "/spatial/*.csv")[0]

    f = open(glob.glob("/group/xulab/Su_Li/Yuzhou_sptl/Data_source/Spatial_Gene_expression1.0.0/Mouse Brain Section (Coronal)" + "/spatial/*.json")[0],)

    image = glob.glob("/group/xulab/Su_Li/Yuzhou_sptl/Data_source/Spatial_Gene_expression1.0.0/Mouse Brain Section (Coronal)" + "/*.tif")[0]

    spatial_vel_full_output = "/group/xulab/Su_Li/Yuzhou_sptl/Processed_data/Spatial_vel/Spatial_Gene_expression1.0.0/Mouse Brain Section (Coronal)" + "_spatial_vel_full.csv"


    #Gene2000 = pd.read_table(glob.glob("./Spatial_Gene_expression1.1.0/" + item + "/*.txt")[0], sep='\t', header=None)
    
    #output_2000 = "/group/xulab/Su_Li/Yuzhou_sptl/Processed_data/Spatial_Gene_expression1.1.0/" + item + "_2000_RGB.csv"
    
    #Exp_2000_output = "/group/xulab/Su_Li/Yuzhou_sptl/Processed_data/Spatial_Gene_expression1.1.0/" + item + "_2000_Exp_only.csv"
    
    # work with velocity_input
    velocity = pd.read_table(velocity_input, sep=',', header=0)
    velocity_new = velocity.rename({'Unnamed: 0':'barcode'}, axis='columns')
    velocity_new['barcode'] = velocity_new['barcode'] + '-1'
    #print(velocity_new.iloc[:10,:12])
    print(velocity_new.shape)

    # Start to work with spatial csv
    
    tissue_positions = pd.read_table(spatial_csv, sep=',', header=None)
    print(f"spatial csv file in {spatial_csv}")
    
    tissue_positions_new = tissue_positions.rename({0:'barcode', 1:'in_tissue', 2:'array_row', 3:'array_col',
                                                    4:'pxl_col_in_fullres', 5:'pxl_row_in_fullres'}, axis='columns')
    print('********tissue_positions_new check********')
    #print(tissue_positions_new.head(6))
    print(tissue_positions_new.columns)
    print(tissue_positions_new.shape)
    
 # Work with .tif and extract RGB
    #
    data = json.load(f)
    # sdf for spot_diameter_fullres
    sdf = data['spot_diameter_fullres']
    f.close()
    rad = round((sdf-1)/2)
    # Read image
    img = cv2.imread(image)
    print(f"image is from {image}")

    # color conversion
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # do image enhancement here
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    equa = cv2.equalizeHist(gray_img)

    cv2.imwrite(os.path.splitext(spatial_vel_full_output)[0]+'_enhancement.png' , equa)

    print('********check img.shape********')
    print(img.shape)

    RGB_list = []
    equa_list = []
    for i in range(len(tissue_positions_new)):
        a = tissue_positions_new['pxl_row_in_fullres'][i]-rad
        b = tissue_positions_new['pxl_row_in_fullres'][i]+rad+1
        c = tissue_positions_new['pxl_col_in_fullres'][i]-rad
        d = tissue_positions_new['pxl_col_in_fullres'][i]+rad+1
        rgb = list(cv2.mean(img[c:d,a:b]))
        equa_gray  = cv2.mean(equa[c:d,a:b])
        #print(rgb)
        RGB_list.append(rgb)

        equa_list.append(equa_gray[0])
        #print(RGB_list)

    tissue_positions_new['RGB'] = RGB_list
    tissue_positions_new['equa'] = equa_list
    tissue_positions_new['R'] = tissue_positions_new['RGB'].str[0]
    tissue_positions_new['G'] = tissue_positions_new['RGB'].str[1]
    tissue_positions_new['B'] = tissue_positions_new['RGB'].str[2]

    tissue_positions_new = tissue_positions_new.drop(columns = 'RGB')
    print('********check tissue_positions_new PLUS RGB********')
    print(tissue_positions_new.head)


    del gray_img, equa, RGB_list, equa_list

    # merge velocity to spatial csv
    spatial_vel_full = pd.merge(tissue_positions_new, velocity_new, how='inner', on='barcode')
    print('********spatial_vel_full check********')
    print(spatial_vel_full.iloc[:10,:12])
    print(spatial_vel_full.iloc[-10:,:12])
    print(spatial_vel_full.shape)
    print(len(spatial_vel_full[spatial_vel_full['in_tissue'] != 0]))
    #print(spatial_vel_full[spatial_vel_full['in_tissue'] != 0].iloc[:10,:10])
    
    # save to csv
    spatial_vel_full.to_csv(spatial_vel_full_output, index=False)
    print(f"spatial_vel_full saved in {spatial_vel_full_output}")
    
#     # Save without RGB information
#     positions_bc_geneName_full.drop(columns=['R', 'G', 'B']).to_csv(full_Exp_output, sep='\t')
#     print(f"Full Expression data saved in {full_Exp_output}")
#
#     # 2000 top variance gene list ### This list provided by Yuzhou, later for different sample, need to specify individually
#
#     #print(Gene2000.head(10))
#
#     list_2000Gene = ['barcode']
#     for item in Gene2000[0]:
#         list_2000Gene.append(item)
# #    #print(list_2000Gene)
#     len(list_2000Gene)
#
#     # only keep 2000 genes's expression, filtered by Yuzhou
#     geneName_bc_df_2000 = geneName_bc_df[list_2000Gene]
#     print('********geneName_bc_df_2000 check********')
#     print(geneName_bc_df_2000.iloc[:10,:10])
#     print(geneName_bc_df_2000.shape)
#
#     # remove duplicates
#     geneName_bc_df_2000_1 = geneName_bc_df_2000.loc[:,~geneName_bc_df_2000.columns.duplicated()]
#     print('********geneName_bc_df_2000_1 remove dupliates check********')
#     print(geneName_bc_df_2000_1.shape)
#
#     # merge geneName_bc_df_2000_1 to spatial csv
#     positions_bc_geneName_2000_1 = pd.merge(tissue_positions_new, geneName_bc_df_2000_1, how='outer', on='barcode')
#     print(positions_bc_geneName_2000_1.iloc[:10,:10])
#     print(positions_bc_geneName_2000_1.iloc[-10:,:10])
#     print(positions_bc_geneName_2000_1.shape)
#     # len(positions_bc_geneName_2000_1[positions_bc_geneName_2000_1['in_tissue'] != 0])
#     # print(positions_bc_geneName_2000_1[positions_bc_geneName_2000_1['in_tissue'] != 0].iloc[:10,:10])
#
#     # save to csv
#
#     positions_bc_geneName_2000_1.to_csv(output_2000, sep='\t')
#     print(f"output saved in {output_2000}")
#
#     # Save without RGB information
#     positions_bc_geneName_2000_1.drop(columns=['R', 'G', 'B']).to_csv(Exp_2000_output, sep='\t')
#     print(f"2000 genes Expression data saved in {Exp_2000_output}")

