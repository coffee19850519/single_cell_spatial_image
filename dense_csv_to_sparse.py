import scipy.sparse as sp_sparse
import numpy as np
import pandas as pd
import os



def dense_csv_to_sparse(densefile, full_output, meta_data, dense_ftype='slide_seq'):
    global start_ind
    positions_bc_gene_name_full = pd.read_table(densefile, sep=',', header=0)
    #print('********positions_bc_gene_name_full check********')
    #print(positions_bc_gene_name_full.iloc[:10, :7])
    #print(positions_bc_gene_name_full.iloc[-10:, :11])
    #print(positions_bc_gene_name_full.shape)
    #print(len(positions_bc_geneName_full[positions_bc_geneName_full['in_tissue'] != 0]))
    #print(positions_bc_geneName_full[positions_bc_geneName_full['in_tissue'] != 0].iloc[:10,:10])

    # only save the spots in tissue
    spots_in_tissue = positions_bc_gene_name_full.loc[positions_bc_gene_name_full['in_tissue'] != 0]
    #print(positions_bc_gene_name_full['in_tissue'].dtype)
    #print(f'len of spots_in_tissue is {len(spots_in_tissue)}')
    #print(spots_in_tissue.iloc[:10, :15])
    #print(spots_in_tissue.iloc[-10:, :15])

    # save to sparse matrix
    if dense_ftype is 'slide_seq':
        start_ind = 9
    elif dense_ftype is '10X':
        start_ind = 10
    spots_in_tissue_arr = spots_in_tissue.iloc[:, start_ind:].to_numpy()
    spots_in_tissue_arr = spots_in_tissue_arr.astype(float)
    #print('********spots_in_tissue_arr check********')
    #print(spots_in_tissue_arr.dtype)
    #print(spots_in_tissue_arr[:start_ind])
    #print(spots_in_tissue_arr.shape)
    spots_in_tissue_spa = sp_sparse.csc_matrix(spots_in_tissue_arr)
    sp_sparse.save_npz(full_output, spots_in_tissue_spa)
    print(f"Fulloutput saved in {full_output}")

    # save the meta data of spots_in_tissue
    spots_in_tissue.iloc[:, :start_ind].to_csv(meta_data, index=False)

input_dir = r'/group/xulab/Su_Li/Yuzhou_sptl/Processed_data/Exp_sparse_mtx/L42523/'
out_dir = r'/group/xulab/Su_Li/Yuzhou_sptl/Processed_data/Exp_sparse_mtx/L42523/test/'


for file in os.listdir(input_dir):
    if os.path.splitext(file)[1] == '.csv':
        densefile = os.path.join(input_dir, file)
        full_output = os.path.join(out_dir, (os.path.splitext(file)[0] + '_spa_full.npz'))
        meta_data = os.path.join(out_dir, (os.path.splitext(file)[0] + '_meta_data.csv'))
        dense_csv_to_sparse(densefile, full_output, meta_data, dense_ftype='slide_seq')

## To use this py, please make "out_dir" and then specify: 1. input_dir; 2. out_dir; 3. dense_ftype = 'slide_seq' or '10X'.
