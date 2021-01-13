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


List_filtered_matrix_h5 = ["Human_Breast_Cancer_Block_A_Section_1",
                           "Human_Breast_Cancer_Block_A_Section_2",
                           "Human_Heart",
                           "Human_Lymph_Node",
                           "Mouse_Brain_Section_Coronal",
                           "Mouse_Brain_Serial_Section_1_Sagittal_Anterior",
                           "Mouse_Brain_Serial_Section_1_Sagittal_Posterior",
                           "Mouse_Brain_Serial_Section_2_Sagittal_Anterior",
                           "Mouse_Brain_Serial_Section_2_Sagittal_Posterior",
                           "Mouse_Kidney_Section_Coronal"]

for item in List_filtered_matrix_h5:
    filtered_matrix_h5 = glob.glob("./Spatial_Gene_expression1.0.0/" + item + "/*.h5")[0]
    
    spatial_csv = glob.glob("./Spatial_Gene_expression1.0.0/" + item + "/spatial/*.csv")[0]
    
    full_output = "/group/xulab/Su_Li/Yuzhou_sptl/Processed_data/Spatial_Gene_expression1.0.0/WithRGB/" + item + "_full_RGB.csv"
    
    full_Exp_output = "/group/xulab/Su_Li/Yuzhou_sptl/Processed_data/Spatial_Gene_expression1.0.0/WithoutRGB/" + item + "_full_Exp_only.csv"
    
    f = open(glob.glob("./Spatial_Gene_expression1.0.0/" + item + "/spatial/*.json")[0],)
    
    image = glob.glob("./Spatial_Gene_expression1.0.0/" + item + "/*.tif")[0]
    
    Gene2000 = pd.read_table(glob.glob("./Spatial_Gene_expression1.0.0/" + item + "/*.txt")[0], sep='\t', header=None)
    
    output_2000 = "/group/xulab/Su_Li/Yuzhou_sptl/Processed_data/Spatial_Gene_expression1.0.0/WithRGB/" + item + "_2000_RGB.csv"
    
    Exp_2000_output = "/group/xulab/Su_Li/Yuzhou_sptl/Processed_data/Spatial_Gene_expression1.0.0/WithoutRGB/" + item + "_2000_Exp_only.csv"
    
    CountMatrix = collections.namedtuple('CountMatrix', ['feature_ref', 'barcodes', 'matrix'])
    
    def get_matrix_from_h5(filename):
        with tables.open_file(filename, 'r') as f:
            mat_group = f.get_node(f.root, 'matrix')
            barcodes = f.get_node(mat_group, 'barcodes').read()
            data = getattr(mat_group, 'data').read()
            indices = getattr(mat_group, 'indices').read()
            indptr = getattr(mat_group, 'indptr').read()
            shape = getattr(mat_group, 'shape').read()
            matrix = sp_sparse.csc_matrix((data, indices, indptr), shape=shape)
    
            feature_ref = {}
            feature_group = f.get_node(mat_group, 'features')
            feature_ids = getattr(feature_group, 'id').read()
            feature_names = getattr(feature_group, 'name').read()
            feature_types = getattr(feature_group, 'feature_type').read()
            feature_ref['id'] = feature_ids
            feature_ref['name'] = feature_names
            feature_ref['feature_type'] = feature_types
            tag_keys = getattr(feature_group, '_all_tag_keys').read()
            feature_ref['genome'] = getattr(feature_group, 'genome').read()
            #for key in tag_keys:
                #feature_ref[key] = getattr(feature_group, key).read()
    
            return CountMatrix(feature_ref, barcodes, matrix)
    
    
    filtered_feature_bc_matrix = get_matrix_from_h5(filtered_matrix_h5)
    dense_matrix = filtered_feature_bc_matrix[2].todense()
    barcodes = filtered_feature_bc_matrix[1]
    feature_genome = filtered_feature_bc_matrix[0]['genome']
    feature_feature_type = filtered_feature_bc_matrix[0]['feature_type']
    feature_id = filtered_feature_bc_matrix[0]['id']
    feature_name = filtered_feature_bc_matrix[0]['name']
    
    print(f"h5 source file name: {filtered_matrix_h5}")
    # transfer dense_matrix to array
    dense_array = np.squeeze(np.asarray(dense_matrix))
    
    # transfer the array to a dataframe
    dense_df = pd.DataFrame(data=dense_array,
                            index=['f'+str(i) for i in range(dense_array.shape[0])],
                           columns=[i for i in range(dense_array.shape[1])])
     
    
    
    # Insert four features
    dense_df.insert(0, "genome", feature_genome, True)
    dense_df.insert(1, "feature_type", feature_feature_type, True)
    dense_df.insert(2, "id", feature_id, True)
    dense_df.insert(3, "name", feature_name, True)
    #print(dense_df.head(6))
    
    # Clean the four features values
    dense_df['genome'] = dense_df['genome'].astype(str)
    dense_df['feature_type'] = dense_df['feature_type'].astype(str)
    dense_df['id'] = dense_df['id'].astype(str)
    dense_df['name'] = dense_df['name'].astype(str)
    dense_df['genome'] = dense_df['genome'].str[2:-1]
    dense_df['feature_type'] = dense_df['feature_type'].str[2:-1]
    dense_df['id'] = dense_df['id'].str[2:-1]
    dense_df['name'] = dense_df['name'].str[2:-1]
    #print(dense_df.iloc[:10, :10])
    
    # Transpose the dataframe
    feature_bc_df = dense_df.T
    #print(feature_bc_df.head(6))
    
    # Need to update barcode array information
    arr0 = np.array([0,0,0,0])
    barcodes_new = np.concatenate((arr0,barcodes),axis=0)
    #print(barcodes_new[:7])
    
    # insert barcodes information
    feature_bc_df.insert(0, "barcode", barcodes_new, True)
    #print(feature_bc_df.iloc[:10, :6])
    
    # and clean barcode
    feature_bc_df['barcode'] = feature_bc_df['barcode'].astype(str)
    feature_bc_df['barcode'] = feature_bc_df['barcode'].str[2:-1]
    #print(feature_bc_df.iloc[:10, :5])
    
    # subset
    geneName_bc_df = feature_bc_df.iloc[4:]
    geneName_bc_df.columns = feature_bc_df.iloc[3]
    geneName_bc_df = geneName_bc_df.rename({list(geneName_bc_df)[0]:'barcode'}, axis='columns')
    print('********geneName_bc_df********')
    print(geneName_bc_df.iloc[:10, :5])
    print(geneName_bc_df.columns)
    print(geneName_bc_df.shape)
    
    # save to csv
    #feature_bc_df.to_csv("/home/lsxgf/data/Yuzhou_Spatial/Data_source/10X-data/HE_stain/human_breast1/human_V1_BC_Feature_bc", sep='\t')
    #geneName_bc_df.to_csv("/home/lsxgf/data/Yuzhou_Spatial/Data_source/10X-data/HE_stain/human_breast1/human_V1_BC_geneName_bc", sep='\t')
    
    # Start to work with spatial csv
    
    tissue_positions = pd.read_table(spatial_csv, sep=',', header=None)
    print(f"spatial csv file in {spatial_csv}")
    
    tissue_positions_new = tissue_positions.rename({0:'barcode', 1:'in_tissue', 2:'array_row', 3:'array_col',
                                                    4:'pxl_col_in_fullres', 5:'pxl_row_in_fullres'}, axis='columns')
    print('********tissue_positions_new check********')
    print(tissue_positions_new.head(6))
    print(tissue_positions_new.columns)
    print(tissue_positions_new.shape)
    
    # Work with .tif and extract RGB
    
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
    
    print('********check img.shape********')
    print(img.shape)
    
    RGB_list = []
    for i in range(len(tissue_positions_new)):
        a = tissue_positions_new['pxl_row_in_fullres'][i]-rad
        b = tissue_positions_new['pxl_row_in_fullres'][i]+rad+1
        c = tissue_positions_new['pxl_col_in_fullres'][i]-rad
        d = tissue_positions_new['pxl_col_in_fullres'][i]+rad+1
        rgb = list(cv2.mean(img[c:d,a:b]))
        #print(rgb)
        RGB_list.append(rgb)
        #print(RGB_list)

    tissue_positions_new['RGB'] = RGB_list
    
    tissue_positions_new['R'] = tissue_positions_new['RGB'].str[0]
    tissue_positions_new['G'] = tissue_positions_new['RGB'].str[1]
    tissue_positions_new['B'] = tissue_positions_new['RGB'].str[2]
    tissue_positions_new = tissue_positions_new.drop(columns = 'RGB')
    print('********check tissue_positions_new PLUS RGB********')
    print(tissue_positions_new.head)
    
    # merge geneName_bc_df to spatial csv
    positions_bc_geneName_full = pd.merge(tissue_positions_new, geneName_bc_df, how='outer', on='barcode')
    print('********positions_bc_geneName_full check********')
    print(positions_bc_geneName_full.iloc[:10,:10])
    print(positions_bc_geneName_full.iloc[-10:,:10])
    print(positions_bc_geneName_full.shape)
    #print(len(positions_bc_geneName_full[positions_bc_geneName_full['in_tissue'] != 0]))
    #print(positions_bc_geneName_full[positions_bc_geneName_full['in_tissue'] != 0].iloc[:10,:10])
    
    # save to csv
    positions_bc_geneName_full.to_csv(full_output, index=False)
    print(f"Fulloutput saved in {full_output}")
    
    # Save without RGB information
    positions_bc_geneName_full.drop(columns=['R', 'G', 'B']).to_csv(full_Exp_output, index=False)
    print(f"Full Expression data saved in {full_Exp_output}")
    
    # 2000 top variance gene list ### This list provided by Yuzhou, later for different sample, need to specify individually 
    
    #print(Gene2000.head(10))
    
    list_2000Gene = ['barcode']
    for item in Gene2000[0]:
        list_2000Gene.append(item)
#    #print(list_2000Gene)
    len(list_2000Gene)

    # only keep 2000 genes's expression, filtered by Yuzhou
    intSet = set(list(geneName_bc_df)) & set(list_2000Gene)
    geneName_bc_df_2000 = geneName_bc_df[list(intSet)]
    print('********geneName_bc_df_2000 check********')
    print(geneName_bc_df_2000.iloc[:10,:10])
    print(geneName_bc_df_2000.shape)
    
    # remove duplicates
    geneName_bc_df_2000_1 = geneName_bc_df_2000.loc[:,~geneName_bc_df_2000.columns.duplicated()]
    print('********geneName_bc_df_2000_1 remove dupliates check********')
    print(geneName_bc_df_2000_1.shape)
    
    # merge geneName_bc_df_2000_1 to spatial csv
    positions_bc_geneName_2000_1 = pd.merge(tissue_positions_new, geneName_bc_df_2000_1, how='outer', on='barcode')
    print(positions_bc_geneName_2000_1.iloc[:10,:10])
    print(positions_bc_geneName_2000_1.iloc[-10:,:10])
    print(positions_bc_geneName_2000_1.shape)
    # len(positions_bc_geneName_2000_1[positions_bc_geneName_2000_1['in_tissue'] != 0])
    # print(positions_bc_geneName_2000_1[positions_bc_geneName_2000_1['in_tissue'] != 0].iloc[:10,:10])
    
    # save to csv
    
    positions_bc_geneName_2000_1.to_csv(output_2000, index=False)
    print(f"output saved in {output_2000}")

    # Save without RGB information
    print("notice this is 'positions_bc_geneName_full'") 
    print(list(positions_bc_geneName_full.drop(columns=['R', 'G', 'B']).columns)[:12])
    print("this is for 'positions_bc_geneName_2000_1'")
    print(list(positions_bc_geneName_2000_1.drop(columns=['R', 'G', 'B']).columns)[:12])
    positions_bc_geneName_2000_1.drop(columns=['R', 'G', 'B']).to_csv(Exp_2000_output, index=False)
    print(f"2000 genes Expression data saved in {Exp_2000_output}")


