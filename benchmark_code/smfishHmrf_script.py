# -*- coding: utf-8 -*-

import sys

import math

import os

import numpy as np

import pandas as pd

import smfishHmrf.reader as reader

from smfishHmrf.HMRFInstance import HMRFInstance

from smfishHmrf.DatasetMatrix import DatasetMatrix, DatasetMatrixSingleField

import smfishHmrf.visualize as visualize

import smfishHmrf.spatial as spatial

import scanpy as sc

import scipy

import scipy.stats

import re

from operator import itemgetter

from scipy.spatial.distance import squareform, pdist

from scipy.stats import percentileofscore

from pathlib import Path

import argparse

from scipy.sparse.csr import csr_matrix

from anndata import AnnData



exp_dir = "/scratch/10x/Human_breast_cancer/sparse_meta_out/"

save_dir = "/scratch/10x/Jixin_script/results/smfishHmrf/"

out_dir = "/scratch/10x/Jixin_script/results/smfishHmrf/temp/"

all_exp_data = os.listdir(exp_dir)



def find_pos_path(exp_dir, sample):

    data_path = exp_dir + "npz/" + "_".join(sample.split("_")[:-1]) + "_metaData.csv"

    position = pd.read_csv(data_path, header=0, index_col=0)

    return position

    

def parse_args():

    parser = argparse.ArgumentParser(description="Consider the combination of parameters in smfishHrmf")

    parser.add_argument('-ngs', type=int, default = 80)

    parser.add_argument('-cutoff', type=float, default = 0.3)

    parser.add_argument('-beta', type=float, default = 9)

    args = parser.parse_args()

    return args



args = parse_args()

ngs = args.ngs

cutoff = args.cutoff

beta = args.beta

parameters = "ngs-" + str(ngs) + "-cutoff-" + str(cutoff) + "-beta-" + str(beta)



for i in all_exp_data:

    if 'npz' in i:

        try:

            # Load spatial locations

            position = pd.read_csv(exp_dir + "_".join(i.split("_")[:-1]) + "_metaData.csv", index_col=0)

            position = position.loc[position.iloc[:, 0] == 1, ]

            

            gene_name = pd.read_csv(exp_dir + "_".join(i.split("_")[:-1]) + "_geneName.csv", index_col=0, header=None).index.values.tolist()

            cell_name = pd.read_csv(exp_dir + "_".join(i.split("_")[:-1]) + "_barcodeID.csv", index_col=0, header=None).index.values.tolist()

            

            # Load spatial data

            current_exp_path = exp_dir + i

            temp_data = np.load(current_exp_path)

            e = csr_matrix((temp_data['data'], temp_data['indices'], temp_data['indptr']), dtype=float).todense()

            # f = AnnData(e)

            # adata = f.transpose()

            # adata.obs.index = cell_name

            # adata.var.index = gene_name

            count_matrix = pd.DataFrame(e, index=gene_name, columns=cell_name).transpose()

            adata = sc.AnnData(count_matrix)
            print(adata.obs.index[:5])

            adata.var_names_make_unique()

            sc.pp.filter_genes(adata, min_cells=1500)

            # sc.pp.filter_cells(adata, min_genes=50)

            count_matrix = count_matrix.loc[adata.obs.index, adata.var.index]

            

            sc.pp.highly_variable_genes(adata, n_top_genes=150)

            t = adata.var.loc[:,'highly_variable'].values == True

            count_matrix = count_matrix.iloc[:, t]

            



            Xcen = position.iloc[:, [1, 2]]

            Xcen = Xcen.values

            barcodes = adata.obs.index

            genes = list(count_matrix.columns)

            field = np.zeros(Xcen.shape[0])

            

            # transpose

            mat_si = count_matrix.values    # Caleculte sihouette

            mat = count_matrix.T.values     # Create DatasetMatrixSingleField object

            

            # create a DatasetMatrixSingleField instance with the input files

            this_dset = DatasetMatrixSingleField(mat, genes, None, Xcen)

            

            # compute neighbor graph (first test various distance cutoff: 0.3%, 0.5%, 1% of lowest pairwise distances)

            this_dset.test_adjacency_list([0.3, 0.5, 0.7, 1], metric="euclidean")

            # use cutoff of 0.3% (gives ~5 neighbors/cell)

            this_dset.calc_neighbor_graph(cutoff, metric="euclidean")

            this_dset.calc_independent_region()

            

            # select genes

            Xcen = Xcen[1:]

            field = field[1:]

            print(Xcen.shape)

            ncell = Xcen.shape[0]

            print("Euclidean distance")

            euc = squareform(pdist(Xcen, metric="euclidean"))

            print("Rank transform")

            dissim = spatial.rank_transform_matrix(euc, reverse=False, rbp_p=0.95)

            res = spatial.calc_silhouette_per_gene(genes=genes, expr=mat_si, dissim=dissim, examine_top=0.1, permutation_test=True, permutations=100)

            

            selected_gene = []

            for j in res[:ngs]:

                temp = j[0]

                selected_gene.append(temp)

            

            outdir = out_dir + "spatial_visium_" + i.split("_")[0] + "_" + parameters

            if not Path(outdir).exists():

                os.mkdir(outdir)

            

            new_dset = this_dset.subset_genes(selected_gene)

            this_hmrf = HMRFInstance("cortex", outdir, new_dset, 7, (beta-0.5, 0.5, 3), tolerance=1e-20)

            this_hmrf.init(nstart=1000, seed=-1)

            this_hmrf.run()

            print("HMRF running fished")

            

            save_name = "_".join(i.split("_")[:-2]) + "_smfishHmrf_LogCPM_" + parameters + ".csv"

            # visualize.domain(this_hmrf, 9, 9.0, dot_size=45, size_factor=10, outfile= save_dir +  i.split("_")[0] + "_smfishHrmf_default.png")

            

            # Save results

            cluster = this_hmrf.domain[(7, beta)]

            result = pd.DataFrame(columns=["barcode", "label"])

            result["barcode"] = barcodes.values

            result["label"] = cluster

            print(result)

            result.to_csv(save_dir + save_name, index=False)

                    

        except Exception as e:

            print("-------------->", e)

    

    


