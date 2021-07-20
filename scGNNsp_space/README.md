# scGNNsp

single cell Graph Neural Networks for spatial

# GAT:

https://github.com/Diego999/pyGAT

train.py
line 111/112:
          "loss= {:.4f}".format(loss_test.data),
          "accuracy= {:.4f}".format(acc_test.data))

Alternative:
https://github.com/gordicaleksa/pytorch-GAT


HGT in dgl
https://github.com/dmlc/dgl/tree/master/examples/pytorch/hgt

original HGT:
https://github.com/acbull/pyHGT



## seqfish_plus in scGNN format, but not use now

# 151673: top 2000 and all non-zero(13796 genes)
python3 -W ignore PreprocessingscGNN.py --datasetName 151673_human_brain_ex.csv --datasetDir 151673/ --LTMGDir 151673/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 151673_human_brain_ex.csv --datasetDir 151673_all/ --LTMGDir 151673_all/ --filetype CSV --cellRatio 1.00 --geneSelectnum 33538

# preprocess
python Preprocessing_IBD_norm.py --datasetName 151507 --method LogCPM
python Preprocessing_IBD_norm.py --datasetName 151508 --method LogCPM
python Preprocessing_IBD_norm.py --datasetName 151509 --method LogCPM
python Preprocessing_IBD_norm.py --datasetName 151510 --method LogCPM
python Preprocessing_IBD_norm.py --datasetName 151669 --method LogCPM
python Preprocessing_IBD_norm.py --datasetName 151670 --method LogCPM
python Preprocessing_IBD_norm.py --datasetName 151671 --method LogCPM
python Preprocessing_IBD_norm.py --datasetName 151672 --method LogCPM
python Preprocessing_IBD_norm.py --datasetName 151673 --method LogCPM
python Preprocessing_IBD_norm.py --datasetName 151674 --method LogCPM
python Preprocessing_IBD_norm.py --datasetName 151675 --method LogCPM
python Preprocessing_IBD_norm.py --datasetName 151676 --method LogCPM

python Preprocessing_IBD_norm.py --datasetName 151507
python Preprocessing_IBD_norm.py --datasetName 151508
python Preprocessing_IBD_norm.py --datasetName 151509
python Preprocessing_IBD_norm.py --datasetName 151510
python Preprocessing_IBD_norm.py --datasetName 151669
python Preprocessing_IBD_norm.py --datasetName 151670
python Preprocessing_IBD_norm.py --datasetName 151671
python Preprocessing_IBD_norm.py --datasetName 151672
python Preprocessing_IBD_norm.py --datasetName 151673
python Preprocessing_IBD_norm.py --datasetName 151674
python Preprocessing_IBD_norm.py --datasetName 151675
python Preprocessing_IBD_norm.py --datasetName 151676

python Preprocessing_IBD_norm.py --datasetName 18-64 --sourcedir /ocean/projects/ccr180012p/shared/image_segmenation/data/10x/Brain_4_ours/normalized_data/
python Preprocessing_IBD_norm.py --datasetName 2-5 --sourcedir /ocean/projects/ccr180012p/shared/image_segmenation/data/10x/Brain_4_ours/normalized_data/
python Preprocessing_IBD_norm.py --datasetName 2-8 --sourcedir /ocean/projects/ccr180012p/shared/image_segmenation/data/10x/Brain_4_ours/normalized_data/
python Preprocessing_IBD_norm.py --datasetName T4857 --sourcedir /ocean/projects/ccr180012p/shared/image_segmenation/data/10x/Brain_4_ours/normalized_data/

python3 -W ignore PreprocessingscGNN.py --datasetName 18-64_human_brain_ex.csv --datasetDir 18-64_sctransform/ --LTMGDir 18-64_sctransform/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None
python3 -W ignore PreprocessingscGNN.py --datasetName 2-5_human_brain_ex.csv --datasetDir 2-5_sctransform/ --LTMGDir 2-5_sctransform/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None
python3 -W ignore PreprocessingscGNN.py --datasetName 2-8_human_brain_ex.csv --datasetDir 2-8_sctransform/ --LTMGDir 2-8_sctransform/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None
python3 -W ignore PreprocessingscGNN.py --datasetName T4857_human_brain_ex.csv --datasetDir T4857_sctransform/ --LTMGDir T4857_sctransform/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None

# all others:
python3 -W ignore PreprocessingscGNN.py --datasetName 151507_human_brain_ex.csv --datasetDir 151507/ --LTMGDir 151507/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 151508_human_brain_ex.csv --datasetDir 151508/ --LTMGDir 151508/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 151509_human_brain_ex.csv --datasetDir 151509/ --LTMGDir 151509/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 151510_human_brain_ex.csv --datasetDir 151510/ --LTMGDir 151510/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 151669_human_brain_ex.csv --datasetDir 151669/ --LTMGDir 151669/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 151670_human_brain_ex.csv --datasetDir 151670/ --LTMGDir 151670/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 151671_human_brain_ex.csv --datasetDir 151671/ --LTMGDir 151671/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 151672_human_brain_ex.csv --datasetDir 151672/ --LTMGDir 151672/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 151673_human_brain_ex.csv --datasetDir 151673/ --LTMGDir 151673/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 151674_human_brain_ex.csv --datasetDir 151674/ --LTMGDir 151674/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 151675_human_brain_ex.csv --datasetDir 151675/ --LTMGDir 151675/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 151676_human_brain_ex.csv --datasetDir 151676/ --LTMGDir 151676/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000

# norm
python3 -W ignore PreprocessingscGNN.py --datasetName 151507_human_brain_ex.csv --datasetDir 151507_scran/ --LTMGDir 151507_scran/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 151508_human_brain_ex.csv --datasetDir 151508_scran/ --LTMGDir 151508_scran/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 151509_human_brain_ex.csv --datasetDir 151509_scran/ --LTMGDir 151509_scran/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 151510_human_brain_ex.csv --datasetDir 151510_scran/ --LTMGDir 151510_scran/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 151669_human_brain_ex.csv --datasetDir 151669_scran/ --LTMGDir 151669_scran/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 151670_human_brain_ex.csv --datasetDir 151670_scran/ --LTMGDir 151670_scran/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 151671_human_brain_ex.csv --datasetDir 151671_scran/ --LTMGDir 151671_scran/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 151672_human_brain_ex.csv --datasetDir 151672_scran/ --LTMGDir 151672_scran/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 151673_human_brain_ex.csv --datasetDir 151673_scran/ --LTMGDir 151673_scran/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 151674_human_brain_ex.csv --datasetDir 151674_scran/ --LTMGDir 151674_scran/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 151675_human_brain_ex.csv --datasetDir 151675_scran/ --LTMGDir 151675_scran/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 151676_human_brain_ex.csv --datasetDir 151676_scran/ --LTMGDir 151676_scran/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000

python3 -W ignore PreprocessingscGNN.py --datasetName 151507_human_brain_ex.csv --datasetDir 151507_sctransform/ --LTMGDir 151507_sctransform/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None
python3 -W ignore PreprocessingscGNN.py --datasetName 151508_human_brain_ex.csv --datasetDir 151508_sctransform/ --LTMGDir 151508_sctransform/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None
python3 -W ignore PreprocessingscGNN.py --datasetName 151509_human_brain_ex.csv --datasetDir 151509_sctransform/ --LTMGDir 151509_sctransform/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None
python3 -W ignore PreprocessingscGNN.py --datasetName 151510_human_brain_ex.csv --datasetDir 151510_sctransform/ --LTMGDir 151510_sctransform/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None
python3 -W ignore PreprocessingscGNN.py --datasetName 151669_human_brain_ex.csv --datasetDir 151669_sctransform/ --LTMGDir 151669_sctransform/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None
python3 -W ignore PreprocessingscGNN.py --datasetName 151670_human_brain_ex.csv --datasetDir 151670_sctransform/ --LTMGDir 151670_sctransform/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None
python3 -W ignore PreprocessingscGNN.py --datasetName 151671_human_brain_ex.csv --datasetDir 151671_sctransform/ --LTMGDir 151671_sctransform/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None
python3 -W ignore PreprocessingscGNN.py --datasetName 151672_human_brain_ex.csv --datasetDir 151672_sctransform/ --LTMGDir 151672_sctransform/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None
python3 -W ignore PreprocessingscGNN.py --datasetName 151673_human_brain_ex.csv --datasetDir 151673_sctransform/ --LTMGDir 151673_sctransform/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None
python3 -W ignore PreprocessingscGNN.py --datasetName 151674_human_brain_ex.csv --datasetDir 151674_sctransform/ --LTMGDir 151674_sctransform/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None
python3 -W ignore PreprocessingscGNN.py --datasetName 151675_human_brain_ex.csv --datasetDir 151675_sctransform/ --LTMGDir 151675_sctransform/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None
python3 -W ignore PreprocessingscGNN.py --datasetName 151676_human_brain_ex.csv --datasetDir 151676_sctransform/ --LTMGDir 151676_sctransform/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None


python -W ignore scGNN.py --datasetName 151673 --datasetDir ./  --outputDir outputdir/ --EM-iteration 2 --Regu-epochs 3 --EM-epochs 3 --quickmode --nonsparseMode --useSpatial --model PAE --useGAEembedding

python -W ignore scGNNsp.py --datasetName 151673 --datasetDir ./  --outputDir outputdir_tw/ --EM-iteration 1 --Regu-epochs 3 --EM-epochs 3 --quickmode --nonsparseMode --useSpatial --model PAE --useGAEembedding --debugMode savePrune --saveinternal --zdim 3 --GAEhidden2 3 --adjtype weighted --prunetype spatialGrid

python -W ignore scGNN.py --datasetName sim --datasetDir ./  --outputDir outputdir/ --quickmode --nonsparseMode --useGAEembedding --useSpatial --model PAE  --PEtype add --resolution 0.5


python -W ignore scGNN.py --datasetName sim1 --datasetDir ./  --outputDir outputdir/ --quickmode --nonsparseMode --useGAEembedding



## usage

# Can try 
pe-type

# encoder/decode mode
python -W ignore scGNN.py --datasetName 151673 --datasetDir ./  --outputDir outputdir_ccG/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --encode-mode cluster --decode-mode cluster --zdim 128 --saveinternal --no-cuda
python -W ignore scGNN.py --datasetName 151673 --datasetDir ./  --outputDir outputdir_c_G/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --encode-mode cluster --zdim 128 --saveinternal --no-cuda
python -W ignore scGNN.py --datasetName 151673 --datasetDir ./  --outputDir outputdir__cG/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --decode-mode cluster --zdim 128 --saveinternal --no-cuda

python -W ignore scGNN.py --datasetName 151673 --datasetDir ./  --outputDir outputdir/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda --zdim 128


python -W ignore scGNN.py --datasetName seqfish_plus --datasetDir ./  --outputDir outputdir_cc/ --nonsparseMode --useSpatial --model PVAE --encode-mode cluster --decode-mode cluster --zdim 128
python -W ignore scGNN.py --datasetName seqfish_plus --datasetDir ./  --outputDir outputdir_c_/ --nonsparseMode --useSpatial --model PVAE --encode-mode cluster --zdim 128
python -W ignore scGNN.py --datasetName seqfish_plus --datasetDir ./  --outputDir outputdir__c/ --nonsparseMode --useSpatial --model PVAE --decode-mode cluster --zdim 128


# 
python -W ignore scGNN.py --datasetName 151673 --datasetDir ./  --outputDir outputdir___G/ --nonsparseMode  --saveinternal --no-cuda --useGAEembedding --resolution 0.8

python -W ignore scGNN.py --datasetName 151673 --datasetDir ./  --outputDir outputdir/ --nonsparseMode --useSpatial --useGAEembedding --encode-mode cluster --decode-mode cluster --saveinternal --no-cuda --quickmode --Regu-epochs 200 --EM-epochs 50

# ARI: 0.2525

python -W ignore scGNN.py --datasetName 151673 --datasetDir ./  --outputDir outputdir/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda --PEtype add


# PE 0.0
python -W ignore scGNN.py --datasetName 151673 --datasetDir ./  --outputDir outputdir/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda --PEtype add --PEalpha 0.0 --pe-type dummy

python -W ignore scGNN.py --datasetName 151673 --datasetDir ./  --outputDir outputdir/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda --PEtype add --PEalpha 10.0

python -W ignore scGNN.py --datasetName 151673 --datasetDir ./  --outputDir outputdir/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda --PEtype add --PEalpha 2.0


# Lots of norm:
do2.sh

python Preprocessing_IBD_norm.py --datasetName 18-64 --sourcedir /ocean/projects/ccr180012p/shared/image_segmenation/data/10x/Brain_4_ours/normalized_data/
python Preprocessing_IBD_norm.py --datasetName 2-5 --sourcedir /ocean/projects/ccr180012p/shared/image_segmenation/data/10x/Brain_4_ours/normalized_data/
python Preprocessing_IBD_norm.py --datasetName 2-8 --sourcedir /ocean/projects/ccr180012p/shared/image_segmenation/data/10x/Brain_4_ours/normalized_data/
python Preprocessing_IBD_norm.py --datasetName T4857 --sourcedir /ocean/projects/ccr180012p/shared/image_segmenation/data/10x/Brain_4_ours/normalized_data/

python3 -W ignore PreprocessingscGNN.py --datasetName 18-64_human_brain_ex.csv --datasetDir 18-64_sctransform/ --LTMGDir 18-64_sctransform/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None
python3 -W ignore PreprocessingscGNN.py --datasetName 2-5_human_brain_ex.csv --datasetDir 2-5_sctransform/ --LTMGDir 2-5_sctransform/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None
python3 -W ignore PreprocessingscGNN.py --datasetName 2-8_human_brain_ex.csv --datasetDir 2-8_sctransform/ --LTMGDir 2-8_sctransform/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None
python3 -W ignore PreprocessingscGNN.py --datasetName T4857_human_brain_ex.csv --datasetDir T4857_sctransform/ --LTMGDir T4857_sctransform/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None

After git pull
do3.sh
python Preprocessing_IBD_norm.py --datasetName 18-64 --sourcedir /ocean/projects/ccr180012p/shared/image_segmenation/data/10x/Brain_4_ours/normalized_data/
python Preprocessing_IBD_norm.py --datasetName 2-5 --sourcedir /ocean/projects/ccr180012p/shared/image_segmenation/data/10x/Brain_4_ours/normalized_data/
python Preprocessing_IBD_norm.py --datasetName 2-8 --sourcedir /ocean/projects/ccr180012p/shared/image_segmenation/data/10x/Brain_4_ours/normalized_data/
python Preprocessing_IBD_norm.py --datasetName T4857 --sourcedir /ocean/projects/ccr180012p/shared/image_segmenation/data/10x/Brain_4_ours/normalized_data/

python3 -W ignore PreprocessingscGNN.py --datasetName 18-64_human_brain_ex.csv --datasetDir 18-64_cpm/ --LTMGDir 18-64_cpm/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None
python3 -W ignore PreprocessingscGNN.py --datasetName 2-5_human_brain_ex.csv --datasetDir 2-5_cpm/ --LTMGDir 2-5_cpm/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None
python3 -W ignore PreprocessingscGNN.py --datasetName 2-8_human_brain_ex.csv --datasetDir 2-8_cpm/ --LTMGDir 2-8_cpm/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None
python3 -W ignore PreprocessingscGNN.py --datasetName T4857_human_brain_ex.csv --datasetDir T4857_cpm/ --LTMGDir T4857_cpm/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None

python Preprocessing_IBD.py --datasetName 18-64 --sourcedir /ocean/projects/ccr180012p/shared/image_segmenation/data/10x/Brain_4_ours/
python Preprocessing_IBD.py --datasetName 2-5 --sourcedir /ocean/projects/ccr180012p/shared/image_segmenation/data/10x/Brain_4_ours/
python Preprocessing_IBD.py --datasetName 2-8 --sourcedir /ocean/projects/ccr180012p/shared/image_segmenation/data/10x/Brain_4_ours/
python Preprocessing_IBD.py --datasetName T4857 --sourcedir /ocean/projects/ccr180012p/shared/image_segmenation/data/10x/Brain_4_ours/

python3 -W ignore PreprocessingscGNN.py --datasetName 18-64_human_brain_ex.csv --datasetDir 18-64/ --LTMGDir 18-64/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 2-5_human_brain_ex.csv --datasetDir 2-5/ --LTMGDir 2-5/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName 2-8_human_brain_ex.csv --datasetDir 2-8/ --LTMGDir 2-8/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000
python3 -W ignore PreprocessingscGNN.py --datasetName T4857_human_brain_ex.csv --datasetDir T4857/ --LTMGDir T4857/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000


## simulation
python3 -W ignore scGNNsp.py --datasetName model1_1 --datasetDir ./  --outputDir outputdirS-model1_1/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda --prunetype spatialGrid --EM-iteration 1 --knn-distance euclidean --PEtypeOp add --pe-type dummy --PEalpha 0.5 --k 8 --pruneTag NA
python3 -W ignore scGNNsp.py --datasetName model2_1 --datasetDir ./  --outputDir outputdirS-model2_1/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda --prunetype spatialGrid --EM-iteration 1 --knn-distance euclidean --PEtypeOp add --pe-type dummy --PEalpha 0.5 --k 8 --pruneTag NA
python3 -W ignore scGNNsp.py --datasetName model3_1 --datasetDir ./  --outputDir outputdirS-model3_1/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda --prunetype spatialGrid --EM-iteration 1 --knn-distance euclidean --PEtypeOp add --pe-type dummy --PEalpha 0.5 --k 8 --pruneTag NA
python3 -W ignore scGNNsp.py --datasetName model4_1 --datasetDir ./  --outputDir outputdirS-model4_1/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda --prunetype spatialGrid --EM-iteration 1 --knn-distance euclidean --PEtypeOp add --pe-type dummy --PEalpha 0.5 --k 8 --pruneTag NA
python3 -W ignore scGNNsp.py --datasetName model5_1 --datasetDir ./  --outputDir outputdirS-model5_1/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda --prunetype spatialGrid --EM-iteration 1 --knn-distance euclidean --PEtypeOp add --pe-type dummy --PEalpha 0.5 --k 8 --pruneTag NA
python3 -W ignore scGNNsp.py --datasetName model6_1 --datasetDir ./  --outputDir outputdirS-model6_1/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda --prunetype spatialGrid --EM-iteration 1 --knn-distance euclidean --PEtypeOp add --pe-type dummy --PEalpha 0.5 --k 8 --pruneTag NA
python3 -W ignore scGNNsp.py --datasetName model7_1 --datasetDir ./  --outputDir outputdirS-model7_1/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda --prunetype spatialGrid --EM-iteration 1 --knn-distance euclidean --PEtypeOp add --pe-type dummy --PEalpha 0.5 --k 8 --pruneTag NA
python3 -W ignore scGNNsp.py --datasetName model8_1 --datasetDir ./  --outputDir outputdirS-model8_1/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda --prunetype spatialGrid --EM-iteration 1 --knn-distance euclidean --PEtypeOp add --pe-type dummy --PEalpha 0.5 --k 8 --pruneTag NA

python3 -W ignore Post_IBD_individual.py --dir /outputdirS-model1_1/ --inputfile model1_1_8_euclidean_NA_dummy_add_0.5_results --datasetname model1_1


python3 -W ignore scGNNsp.py --datasetName model1_1 --datasetDir ./  --outputDir outputdir-model1_1/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda  --knn-distance euclidean --PEtypeOp add --pe-type dummy --PEalpha 0.5 --k 8 --pruneTag NA
python3 -W ignore scGNNsp.py --datasetName model2_1 --datasetDir ./  --outputDir outputdir-model2_1/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda  --knn-distance euclidean --PEtypeOp add --pe-type dummy --PEalpha 0.5 --k 8 --pruneTag NA
python3 -W ignore scGNNsp.py --datasetName model3_1 --datasetDir ./  --outputDir outputdir-model3_1/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda  --knn-distance euclidean --PEtypeOp add --pe-type dummy --PEalpha 0.5 --k 8 --pruneTag NA
python3 -W ignore scGNNsp.py --datasetName model4_1 --datasetDir ./  --outputDir outputdir-model4_1/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda  --knn-distance euclidean --PEtypeOp add --pe-type dummy --PEalpha 0.5 --k 8 --pruneTag NA
python3 -W ignore scGNNsp.py --datasetName model5_1 --datasetDir ./  --outputDir outputdir-model5_1/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda  --knn-distance euclidean --PEtypeOp add --pe-type dummy --PEalpha 0.5 --k 8 --pruneTag NA
python3 -W ignore scGNNsp.py --datasetName model6_1 --datasetDir ./  --outputDir outputdir-model6_1/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda  --knn-distance euclidean --PEtypeOp add --pe-type dummy --PEalpha 0.5 --k 8 --pruneTag NA
python3 -W ignore scGNNsp.py --datasetName model7_1 --datasetDir ./  --outputDir outputdir-model7_1/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda  --knn-distance euclidean --PEtypeOp add --pe-type dummy --PEalpha 0.5 --k 8 --pruneTag NA
python3 -W ignore scGNNsp.py --datasetName model8_1 --datasetDir ./  --outputDir outputdir-model8_1/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda  --knn-distance euclidean --PEtypeOp add --pe-type dummy --PEalpha 0.5 --k 8 --pruneTag NA

python3 -W ignore Post_IBD_individual.py --dir /outputdir-model1_1/ --inputfile model1_1_8_euclidean_NA_dummy_add_0.5_results --datasetname model1_1


python3 -W ignore scGNNsp.py --datasetName model1_1 --datasetDir ./  --outputDir outputdirMV-model1_1/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda --debugMode savePrune --EM-iteration 1 --useMultiView  --knn-distance euclidean --PEtypeOp add --pe-type geom_lowf --PEalpha 0.0 --k 10 --pruneTag STD
python3 -W ignore scGNNsp.py --datasetName model2_1 --datasetDir ./  --outputDir outputdirMV-model2_1/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda --debugMode savePrune --EM-iteration 1 --useMultiView  --knn-distance euclidean --PEtypeOp add --pe-type geom_lowf --PEalpha 0.0 --k 10 --pruneTag STD
python3 -W ignore scGNNsp.py --datasetName model3_1 --datasetDir ./  --outputDir outputdirMV-model3_1/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda --debugMode savePrune --EM-iteration 1 --useMultiView  --knn-distance euclidean --PEtypeOp add --pe-type geom_lowf --PEalpha 0.0 --k 10 --pruneTag STD
python3 -W ignore scGNNsp.py --datasetName model4_1 --datasetDir ./  --outputDir outputdirMV-model4_1/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda --debugMode savePrune --EM-iteration 1 --useMultiView  --knn-distance euclidean --PEtypeOp add --pe-type geom_lowf --PEalpha 0.0 --k 10 --pruneTag STD
python3 -W ignore scGNNsp.py --datasetName model5_1 --datasetDir ./  --outputDir outputdirMV-model5_1/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda --debugMode savePrune --EM-iteration 1 --useMultiView  --knn-distance euclidean --PEtypeOp add --pe-type geom_lowf --PEalpha 0.0 --k 10 --pruneTag STD
python3 -W ignore scGNNsp.py --datasetName model6_1 --datasetDir ./  --outputDir outputdirMV-model6_1/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda --debugMode savePrune --EM-iteration 1 --useMultiView  --knn-distance euclidean --PEtypeOp add --pe-type geom_lowf --PEalpha 0.0 --k 10 --pruneTag STD
python3 -W ignore scGNNsp.py --datasetName model7_1 --datasetDir ./  --outputDir outputdirMV-model7_1/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda --debugMode savePrune --EM-iteration 1 --useMultiView  --knn-distance euclidean --PEtypeOp add --pe-type geom_lowf --PEalpha 0.0 --k 10 --pruneTag STD
python3 -W ignore scGNNsp.py --datasetName model8_1 --datasetDir ./  --outputDir outputdirMV-model8_1/ --nonsparseMode --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda --debugMode savePrune --EM-iteration 1 --useMultiView  --knn-distance euclidean --PEtypeOp add --pe-type geom_lowf --PEalpha 0.0 --k 10 --pruneTag STD



# script
labelname = 'T4857/label.csv'
df = pd.read_csv(labelname)
listBench = df['layer'].to_numpy().tolist()
set(listBench)

.vscode
                "--datasetName","model1_1",
                "--datasetDir","./",
                "--outputDir","outputdir_test/",
                "--resolution","0.3",
                "--nonsparseMode","--useSpatial",
                "--model","PAE","--useGAEembedding",
                "--saveinternal",
                "--no-cuda",
                "--knn-distance","correlation",
                "--PEtypeOp","add",
                "--pe-type","dummy",
                "--PEalpha","0.5",
                "--k","8",
                "--pruneTag","STD",
                "--useMultiView"