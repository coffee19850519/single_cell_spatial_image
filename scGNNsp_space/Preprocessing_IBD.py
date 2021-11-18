import numpy as np
import pandas as pd
import os
import argparse
current_path = os.path.abspath('.')

# The data is from:
# http://research.libd.org/spatialLIBD/
parser = argparse.ArgumentParser(description='Preprocessing IBD data')
parser.add_argument('--datasetName', type=str, default='151673',
                    help='slice name')  
parser.add_argument('--sourcedir', type=str, default='/ocean/projects/ccr180012p/shared/image_segmenation/data/10x/Human_brain_JH_12/',
                    help='directory')                   
args = parser.parse_args()

fileFolder = current_path+'/'+args.datasetName+'/'
if not os.path.exists(fileFolder):
    os.makedirs(fileFolder)

filename = args.sourcedir+args.datasetName+'_human_brain.csv'
outfilename = fileFolder+args.datasetName+'_human_brain_ex.csv'
coordsfilename = fileFolder+'coords_array.npy'
labelname = fileFolder+'label.csv'

df = pd.read_csv(filename)
filter_col = [col for col in df if col.startswith('ENSG') or col.startswith('barcode')]
dfEX = df[filter_col]
dfEX= dfEX.T
dfEX.to_csv(outfilename,header=False)

llist = ['barcode','layer']
dfLabel = df[llist]
dfLabel.to_csv(labelname)


# coordinates
coords_list = []
for i in range(df.shape[0]):
    coords_list.append((df.iloc[i]['array_row'],df.iloc[i]['array_col']))

np.save(coordsfilename,np.array(coords_list))
