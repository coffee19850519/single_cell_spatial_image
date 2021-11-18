import numpy as np
import pandas as pd
import os
import argparse
current_path = os.path.abspath('.')

# The data is from:
# http://research.libd.org/spatialLIBD/
# Use Preprocessing_IBD.py to get the cooridinates first, then proceed to the normalized expression
parser = argparse.ArgumentParser(description='Preprocessing IBD data for normalization')
parser.add_argument('--datasetName', type=str, default='151673',
                    help='slice name')  
parser.add_argument('--sourcedir', type=str, default='/ocean/projects/ccr180012p/shared/image_segmenation/data/10x/Human_brain_JH_12/normalized_data/',
                    help='directory')
parser.add_argument('--method', type=str, default='scran',
                    help='directory')                   
args = parser.parse_args()

# fileFolder = current_path+'/'+args.datasetName+'_'+args.method+'/'
# fileFolder = current_path+'/'+args.datasetName+'_sctransform/'
# fileFolder = current_path+'/'+args.datasetName+'_cpm/'
fileFolder = current_path+'/'+args.datasetName+'F_cpm/'
if not os.path.exists(fileFolder):
    os.makedirs(fileFolder)

# filename = args.sourcedir+args.method+'/'+args.datasetName+'_humanBrain_'+args.method+'.csv'
# filename = args.sourcedir+'LogCPM/'+args.datasetName+'_humanBrain_cpm.csv'
filename = args.sourcedir+'LogCPM/'+args.datasetName+'_humanBrain_LogCPM_den.csv'
# filename = args.sourcedir+'scTransform/'+args.datasetName+'_humanBrain_sctranform.csv'
outfilename = fileFolder+args.datasetName+'_human_brain_ex.csv'


df = pd.read_csv(filename)
dfEX= df.T
dfEX.to_csv(outfilename,header=False)

