import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def preprocessingCSV(anndata, delim='comma', transform='log', cellRatio=0.99, geneRatio=0.99, geneCriteria='variance', geneSelectnum=2000, transpose=False, tabuCol=''):
    '''
    preprocessing CSV files:
    transform='log' or 'None'
    '''
    # expressionFilename = dir + datasetName
    # if not os.path.exists(expressionFilename):
    #     print('Dataset ' + expressionFilename + ' not exists!')
    #
    # print('Input scRNA data in CSV format is validated, start reading...')

    tabuColList = []
    tmplist = tabuCol.split(",")
    for item in tmplist:
        tabuColList.append(item)
    index_col = []
    # print('---------------anndata.X.A.T----------------',anndata.X.A.T.shape[1])
    for i in range(0,anndata.X.A.T.shape[1]):
        index_col.append(i)


    df = pd.DataFrame()
    df = anndata.X.A.T

    print('Data loaded, start filtering...')
    if transpose == True:
        df = df.T
    # df.columns.name = index_col
    df0 = pd.DataFrame(df,columns=index_col)
    df = df0

    df1 = df[df.astype('bool').mean(axis=1) >= (1-geneRatio)]

    print('After preprocessing, {} genes remaining'.format(df1.shape[0]))
    criteriaGene = df1.astype('bool').mean(axis=0) >= (1-cellRatio)
    df2 = df1[df1.columns[criteriaGene]]########
    print('After preprocessing, {} cells have {} nonzero'.format(
        df2.shape[1], geneRatio))
    criteriaSelectGene = df2.var(axis=1).sort_values()[-geneSelectnum:]
    df3 = df2.loc[criteriaSelectGene.index]
    if transform == 'log':
        df3 = df3.transform(lambda x: np.log(x + 1))
    #df3 is Use_expression.csv
    # df3.to_csv(csvFilename)
    print('---------------------Use_expression---------------------------')

    return df3

def generate_coords_sc(anndata, sample, scgnnsp_dist, scgnnsp_alpha, scgnnsp_k, scgnnsp_zdim, scgnnsp_bypassAE):

    coords_list = [list(t) for t in zip(anndata.obs["array_row"].tolist(), anndata.obs["array_col"].tolist())]
    coords = np.array(coords_list)

    ues_expression = preprocessingCSV(anndata, 'comma', None, 1.00, 0.99, 'variance', 2000, False,'')
    # print("Preprocessing Done. Total Running Time: %s seconds" %
    #       (time.time() - start_time))


    return coords, ues_expression










