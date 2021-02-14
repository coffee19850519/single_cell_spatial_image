import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as pairwise
import os
from tqdm import tqdm


def neighbor_cell_list(cell_data, cut_off_radius = 5):
    # Calculate dist_matrix
    coors = cell_data[['array_row','array_col']].values
    dist_matrix = pairwise.pairwise_distances(coors)
    neighbor_index = np.nonzero(dist_matrix <= cut_off_radius)
    del dist_matrix
    return neighbor_index

def merge_neighboring_cells(cell_data_file, feature_start_idx, cut_off_radius):
    cell_data = pd.read_csv(cell_data_file)
    neighbor_index = neighbor_cell_list(cell_data, cut_off_radius)
    # Construct tour
    visited_cells = []
    #declear a empty dataframe to save the remaining cell data
    remained_cells = pd.DataFrame(columns= cell_data.columns.values.tolist())
    for i in tqdm(range(0, len(cell_data))):
        if i not in visited_cells:
            #retrieve all its neighbors
            current_neighbor_indics = neighbor_index[1][neighbor_index[0] == i]
            #merge all neighbors by mean of neighbor features
            merged_features = cell_data.iloc[current_neighbor_indics, feature_start_idx:].sum()
            center_cell_meta_data =  cell_data.iloc[i, :feature_start_idx]

            remained_cells = remained_cells.append([pd.concat([center_cell_meta_data, merged_features], axis= 0)], ignore_index= True)

            visited_cells.extend(current_neighbor_indics)
            del merged_features, center_cell_meta_data
        else:
            continue
    remained_cells.to_csv(os.path.splitext(file_path)[0] + '_merged.csv', index=False)
    del visited_cells, cell_data, remained_cells, neighbor_index


if __name__ == '__main__':
    file_path = r'/home/fei/Desktop/sub_matrix_slide_seq_gene_expression_1000_cell.csv'
    '''
    file_path: expression file at cell level
    feature_start_idx: the gene start index in expression file
    cut_off_radius: defined the maximum neighboring radius
    '''
    merge_neighboring_cells(file_path, feature_start_idx= 9, cut_off_radius=20)
