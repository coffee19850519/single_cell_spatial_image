import numpy as np
import pandas as pd
import os
import argparse
current_path = os.path.abspath('.')

# For four ours data, run on macbook
# name = '18-64'
# name = '2-5'
# name = '2-8'
name = 'T4857'
# filename = '/Users/wangjue/Documents/results_scGNNsp/16_data_benchmark/'+name+'_benchmark.csv'
filename = '/ocean/projects/ccr180012p/shared/image_segmenation/data/10x/new_4/sparse_meta_out/'+name+'_humanBrain_metaData.csv'

coordsfilename = name+'F_cpm/coords_array.npy'
labelname = name+'F_cpm/label.csv'

df = pd.read_csv(filename)
# filter_col = [col for col in df if col.startswith('ENSG') or col.startswith('barcode')]
# dfEX = df[filter_col]
# dfEX= dfEX.T
# dfEX.to_csv(outfilename,header=False)

llist = ['barcode','Layers']
dfLabel = df[llist]
dfLabel.to_csv(labelname)


# coordinates
coords_list = []
for i in range(df.shape[0]):
    coords_list.append((df.iloc[i]['array_row'],df.iloc[i]['array_col']))

np.save(coordsfilename,np.array(coords_list))
