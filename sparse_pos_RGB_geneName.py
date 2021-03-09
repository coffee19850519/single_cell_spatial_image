import collections
import scipy.sparse as sp_sparse
import tables
import numpy as np
import pandas as pd
import os
import glob
import cv2
import json


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


def sparse_to_dense_gene_name_bc_df(filtered_matrix_h5):
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

    # Insert gene name feature
    dense_df.insert(0, "name", feature_name, True)
    #print(dense_df.head(6))

    # Clean the gene name feature
    dense_df['name'] = dense_df['name'].astype(str)
    dense_df['name'] = dense_df['name'].str[2:-1]
    # #print(dense_df.iloc[:10, :10])

    # Transpose the dataframe
    feature_bc_df = dense_df.T
    #print(feature_bc_df.head(6))

    # Update barcode array information
    arr0 = np.array([0])
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
    gene_name_bc_df = feature_bc_df.iloc[1:]
    gene_name_bc_df.columns = feature_bc_df.iloc[0]
    gene_name_bc_df = gene_name_bc_df.rename({list(gene_name_bc_df)[0]:'barcode'}, axis='columns')
    print('********gene_name_bc_df********')
    print(gene_name_bc_df.iloc[:10, :5])
    print(gene_name_bc_df.columns)
    print(gene_name_bc_df.shape)

    return gene_name_bc_df
# save to csv
#feature_bc_df.to_csv("/home/lsxgf/data/Yuzhou_Spatial/Data_source/10X-data/HE_stain/human_breast1/human_V1_BC_Feature_bc", sep='\t')
#geneName_bc_df.to_csv("/home/lsxgf/data/Yuzhou_Spatial/Data_source/10X-data/HE_stain/human_breast1/human_V1_BC_geneName_bc", sep='\t')

# Start to work with spatial csv


def gen_spa_meta(filtered_matrix_h5, spatial_csv, spatial_json_file, image, full_output, meta_data, panel_gene_file, panel_output):
    tissue_positions = pd.read_table(spatial_csv, sep=',', header=None)
    print(f"spatial csv file in {spatial_csv}")

    tissue_positions_new = tissue_positions.rename({0:'barcode', 1:'in_tissue', 2:'array_row', 3:'array_col',
                                                4:'pxl_col_in_fullres', 5:'pxl_row_in_fullres'}, axis='columns')
    print('********tissue_positions_new check********')
    print(tissue_positions_new.head(6))
    print(tissue_positions_new.columns)
    print(tissue_positions_new.shape)

    # Work with .tif and extract RGB

    data = json.load(spatial_json_file)
    # sdf for spot_diameter_fullres
    sdf = data['spot_diameter_fullres']
    spatial_json_file.close()
    rad = round((sdf-1)/2)

    # Read image
    img = cv2.imread(image)
    print(f"image is from {image}")

    # color conversion
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # do image enhancement here
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    equa = cv2.equalizeHist(gray_img)

    # Generate enhanced grey figure
    #cv2.imwrite(os.path.splitext(full_output)[0][:-9]+'_enhancement.png', equa)

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

    # merge geneName_bc_df to spatial csv
    gene_name_bc_df = sparse_to_dense_gene_name_bc_df(filtered_matrix_h5)
    positions_bc_gene_name_full = pd.merge(tissue_positions_new, gene_name_bc_df, how='outer', on='barcode')
    print('********positions_bc_gene_name_full check********')
    print(positions_bc_gene_name_full.iloc[:10, :7])
    print(positions_bc_gene_name_full.iloc[-10:, :11])
    print(positions_bc_gene_name_full.shape)
    #print(len(positions_bc_geneName_full[positions_bc_geneName_full['in_tissue'] != 0]))
    #print(positions_bc_geneName_full[positions_bc_geneName_full['in_tissue'] != 0].iloc[:10,:10])

    # only save the spots in tissue
    spots_in_tissue = positions_bc_gene_name_full.loc[positions_bc_gene_name_full['in_tissue'] != 0]
    #print(positions_bc_gene_name_full['in_tissue'].dtype)
    #print(f'len of spots_in_tissue is {len(spots_in_tissue)}')
    #print(spots_in_tissue.iloc[:10, :15])
    #print(spots_in_tissue.iloc[-10:, :15])

    # save to sparse matrix
    spots_in_tissue_arr = spots_in_tissue.iloc[:, 10:].to_numpy()
    spots_in_tissue_arr = spots_in_tissue_arr.astype(float)
    print('********spots_in_tissue_arr check********')
    print(spots_in_tissue_arr.dtype)
    print(spots_in_tissue_arr[:10])
    print(spots_in_tissue_arr.shape)
    spots_in_tissue_spa = sp_sparse.csc_matrix(spots_in_tissue_arr)
    sp_sparse.save_npz(full_output, spots_in_tissue_spa)
    print(f"Fulloutput saved in {full_output}")

    # save the meta data of spots_in_tissue
    spots_in_tissue.iloc[:, :10].to_csv(meta_data, index=False)

    # save the column names
    with open(os.path.join(out_dir, (os.path.splitext(file)[0] + '_gene_name.txt')), 'w') as f:
        for item in list(gene_name_bc_df)[10:]:
            f.write("%s\n" % item)
        f.close()

    del spots_in_tissue_arr, positions_bc_gene_name_full

    if panel_gene_file is not None:
        panel_gene = pd.read_table(panel_gene_file, sep='\t', header=None)
        list_panel_gene = ['barcode']
        for item in panel_gene[0]:
            list_panel_gene.append(item)
        #    #print(list_panel_gene)
        len(list_panel_gene)

        # only keep panel gene's expression, filtered by Yuzhou
        intset = set(list(gene_name_bc_df)) & set(list_panel_gene)
        gene_name_bc_df_panel = gene_name_bc_df[list(intset)]
        print('********gene_name_bc_df_panel check********')
        print(gene_name_bc_df_panel.iloc[:10, :10])
        print(gene_name_bc_df_panel.shape)

        # remove duplicates
        gene_name_bc_df_panel_1 = gene_name_bc_df_panel.loc[:, ~gene_name_bc_df_panel.columns.duplicated()]
        print('********gene_name_bc_df_panel_1 remove dupliates check********')
        print(gene_name_bc_df_panel_1.shape)

        # merge geneName_bc_df_2000_1 to spatial csv
        positions_bc_gene_name_panel_1 = pd.merge(tissue_positions_new, gene_name_bc_df_panel_1, how='outer', on='barcode')
        print(positions_bc_gene_name_panel_1.iloc[:10,:10])
        print(positions_bc_gene_name_panel_1.iloc[-10:,:10])
        print(positions_bc_gene_name_panel_1.shape)
        # len(positions_bc_geneName_2000_1[positions_bc_geneName_2000_1['in_tissue'] != 0])
        # print(positions_bc_geneName_2000_1[positions_bc_geneName_2000_1['in_tissue'] != 0].iloc[:10,:10])

        # only save in tissue to sparse matrix
        panel_spots_in_tissue = positions_bc_gene_name_panel_1.loc[positions_bc_gene_name_panel_1['in_tissue'] != 0]
        panel_spots_in_tissue_spa = sp_sparse.csc_matrix(panel_spots_in_tissue.iloc[:, 10:].to_numpy().astype(float))
        sp_sparse.save_npz(panel_output, panel_spots_in_tissue_spa)
        print(f"output saved in {panel_output}")



input_dir = r'/group/xulab/Su_Li/Yuzhou_sptl/Data_source/Spatial_Gene_expression1.2.0/'
out_dir = r'/group/xulab/Su_Li/Yuzhou_sptl/Processed_data/Exp_sparse_mtx/Spatial_Gene_expression1.2.0/'


for file in os.listdir(input_dir):
    # here it has strict requirement of the structure of the files. It need to be used by broad user,
    # need to set those files carefully.
    filtered_matrix_h5 = glob.glob(input_dir + file + "/*.h5")[0]
    spatial_csv = glob.glob(input_dir + file + "/spatial/*.csv")[0]
    spatial_json_file = open(glob.glob(input_dir + file + "/spatial/*.json")[0],)
    image = glob.glob(input_dir + file + "/*.tif")[0]
    full_output = os.path.join(out_dir, (os.path.splitext(file)[0] + '_spa_full.npz'))
    meta_data = os.path.join(out_dir, (os.path.splitext(file)[0] + '_meta_data.csv'))
    # check if panel gene list is provided
    try:
        panel_gene_file = glob.glob(input_dir + file + "/*.txt")[0]
    except IndexError:
        panel_gene_file = None
        panel_output = None
    else:
        panel_output = os.path.join(out_dir, (os.path.splitext(file)[0] + '_spa_panel.npz'))

    gen_spa_meta(filtered_matrix_h5, spatial_csv, spatial_json_file, image, full_output, meta_data, panel_gene_file, panel_output)

