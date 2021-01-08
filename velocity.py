import scvelo as scv
import scanpy as sc
import pandas as pd
import argparse

# pip install python-igraph
# pip install louvain

# Use scvelo to infer RNAvelocity
# Ref:https://scvelo.readthedocs.io/VelocityBasics.html

# scv.logging.print_version()
# scv.settings.verbosity = 3  # show errors(0), warnings(1), info(2), hints(3)
# scv.settings.presenter_view = True  # set max width size for presenter view
# scv.set_figure_params('scvelo')  # for beautified visualization

parser = argparse.ArgumentParser(description='main benchmark for scRNA with timer and mem')
parser.add_argument('--bamName', type=str, default='V1_Mouse_Brain_Sagittal_Anterior_possorted_genome_bam_8GH6I',
                    help='bamName')
parser.add_argument('--inputDir', type=str, default='/storage/hpc/group/xulab/wangjue/spatialData/out/',
                    help='inputDir')
parser.add_argument('--outputDir', type=str, default='/storage/hpc/group/xulab/wangjue/spatialData/RNA_Velocity/',
                    help='bamName')
args = parser.parse_args()

bamName = args.bamName
# Debug in windows
# inputDir = 'C:\\Users\\wangjue.UMC-USERS\\Desktop\\scGNN\\'
inputDir = args.inputDir
outputDir = args.outputDir
adata = scv.read(inputDir+bamName, cache=True)
adata.var_names_make_unique()

# scv.pl.proportions(adata)

# scv.pp.filter_and_normalize(adata, min_shared_counts=20)
scv.pp.filter_and_normalize(adata)
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

scv.tl.velocity(adata)
scv.tl.velocity_graph(adata)
indexList = adata.obs.index.tolist()
outindexList = []
for item in indexList:
    outindexList.append(item.split(':')[1])

df = pd.DataFrame(adata.layers['velocity'],index=outindexList,columns=adata.var['Accession'].tolist())
df.to_csv(outputDir+bamName+'.csv')

# sc.tl.louvain(adata)

# scv.tl.umap(adata)
# scv.pl.velocity_embedding_stream(adata, basis='umap')
# scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, dpi=120)
# scv.pl.velocity_graph(adata, threshold=.1)

##tracing
# x, y = scv.utils.get_cell_transitions(adata, basis='umap', starting_cell=70)
# ax = scv.pl.velocity_graph(adata, c='lightgrey', edge_width=.05, show=False)
# ax = scv.pl.scatter(adata, x=x, y=y, s=120, c='ascending', cmap='gnuplot', ax=ax)
##pseudotime
# scv.tl.velocity_pseudotime(adata)
# scv.pl.scatter(adata, color='velocity_pseudotime', cmap='gnuplot')

#Markov transition matrix
# scv.utils.get_transition_matrix(adata)
