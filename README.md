[docs-image]: https://readthedocs.org/projects/pytorch-geometric/badge/?version=latest
[docs-url]: https://resept-last.readthedocs.io/en/latest/Case%20study.html

# Define and visualize pathological architectures of human tissues from spatially resolved transcriptomics using deep learning
**[Paper](https://www.biorxiv.org/content/10.1101/2021.07.08.451210v1)** 
 
A novel method to reconstruct an RGB image of spots using the sequencing data from spatially resolved transcriptomics to identify spatial context and functional zonation.

<p align="center">
  <img height="300" width="700" src="https://github.com/yuyang-0825/image/blob/main/figure1.png" />
</p>


 
## System Requirements

### Hardware Requirements
 
``` RESEPT ``` package requires a standard computer with enough RAM to support the in-memory operations. A GPU with enough VRAM or multi-core CPU is recommended for acceleration.

### Software Requirements

#### OS Requirements
This package is supported for Linux. The package has been tested on the following systems:
* Linux: Ubuntu 20.04

#### Python Dependencies
``` RESEPT ``` mainly depends on the Python scientific stack.
```
numpy 1.18.1
torch 1.4.0
networkx 2.4
pandas 0.25.3
matplotlib 3.1.2
seaborn 0.9.0
umap-learn 0.3.10
munkres 1.1.2
tqdm 4.48.0
python-igraph 0.8.3
scanpy 1.7.2
scikit-image 0.18.1
opencv-python 4.5.1.48
louvain  0.7.0
anndata 0.7.6
mmcv-full  1.3.0
mmsegmentation 0.12.0
```

## Installation Guide

### Install dependency packages
```
pip install numpy 
pip install networkx 
pip install pandas 
pip install matplotlib 
pip install seaborn 
pip install umap-learn 
pip install munkres 
pip install tqdm 
pip install python-igraph 
pip install scanpy 
pip install scikit-image
pip install opencv-python 
pip install louvain  
pip install anndata
pip install mmcv-full  
```
This takes 20-25 mins to install all dependencies.

### Install RESEPT from GitHub
```
git clone https://github.com/OSU-BMBL/RESEPT
cd RESEPT
```
## Demo
### Evaluate tissue architecture with annotations
Run the following command line to generate RGB images from different embedding parameters, segmentation maps with top5 Moran's I and their evaluation metrics.
Please download the corresponding pre-trained model from [click here for downloading data and model](https://bmbl.bmi.osumc.edu/downloadFiles/data_and_model/data_and_model.zip) and put it in the specified folder.
```
wget https://bmbl.bmi.osumc.edu/downloadFiles/RESEPT/RESEPT.zip 
unzip RESEPT.zip
python evaluation_pipeline.py -matrix Demo/S13/S13_filtered_feature_bc_matrix.h5  -csv Demo/S13/spatial/tissue_positions_list.csv  -json Demo/S13/spatial/scalefactors_json.json -out Demo_result  -method scGNN  -transform logcpm -label Demo/S13.csv -checkpoint Demo/checkpoint/S13_scGNN.pth
```

#### Command Line Arguments:
*	-matrix specify path for raw gene expression data provided by 10X in h5 file. [type:str]
*	-csv specify path for meta file recording tissue positions provided by 10x in csv file. [type:str]
*	-json specify path for scale factors provided by 10x in json file. [type:str]
*	-label specify path for annotation file recording cell barcodes and their annotations, which is used for calculating ARI. [type:str]
*	-checkpoint specify path for pretrained model file. [type:str]
*	-out specify output root folder. [type:str]
*	-method specify embedding method in use: scGNN or spaGCN. [type:str]
*	-transform specify data pre-transform: log, logcpm or None. [type:str]

#### Expected Results
RESEPT stores the generative results in the following structure:
   ```
      Demo_result/
      |__RGB_images/
      |__segmentation_evaluation/
            |__segmentation_map/
            |__top5_evaluation.csv
   ```
*	-The folder 'RGB_images' stores generative RGB images from different embedding parameters. 
*	-The folder 'segmentation_map' stores visuals of segmentation results with top5 Moran's I. 
*	-The file 'top5_evaluation.csv' records various evaluation metrics corresponding to segmentation results with top5 Moran's I .
*	-This Demo takes 30-35 mins to generate all results on a machine with a multi-core CPU.

### predict tissue architecture without annotation
Run the following command line to generate RGB images from different embedding parameters, segmentation maps with top5 Moran's I and their Moran's I value.
Please download the corresponding pre-trained model from [click here for downloading data and model](https://bmbl.bmi.osumc.edu/downloadFiles/data_and_model/data_and_model.zip) and put it in the specified folder.
```
wget https://bmbl.bmi.osumc.edu/downloadFiles/RESEPT/RESEPT.zip 
unzip RESEPT.zip
python test_pipeline.py -matrix Demo/S13/S13_filtered_feature_bc_matrix.h5  -csv Demo/S13/spatial/tissue_positions_list.csv  -json Demo/S13/spatial/scalefactors_json.json -out Demo_result  -method scGNN  -transform logcpm -checkpoint Demo/checkpoint/S13_scGNN.pth
```

#### Command Line Arguments:
*	-matrix specify path for raw gene expression data provided by 10X in h5 file. [type:str]
*	-csv specify path for meta file recording tissue positions provided by 10x in csv file. [type:str]
*	-json specify path for scale factors provided by 10x in json file. [type:str]
*	-checkpoint specify path for pretrained model file. [type:str]
*	-out specify output root folder. [type:str]
*	-method specify embedding method in use: scGNN or spaGCN. [type:str]
*	-transform specify data pre-transform: log, logcpm or None. [type:str]

#### Expected Results
RESEPT stores the generative results in the following structure:
   ```
      Demo_result/
      |__RGB_images/
      |__segmentation_test/
            |__segmentation_map/
            |__top5_MI_value.csv
   ```
*	-The folder 'RGB_images' stores generative RGB images from different embedding parameters. 
*	-The folder 'segmentation_map' stores visuals of segmentation results with top5 Moran's I. 
*	-The file 'top5_MI_value.csv' records Moran's I value corresponding to segmentation results with top5 Moran's I.
*	-This Demo takes 30-35 mins to generate all results on a machine with a multi-core CPU.

 
### Training pipeline
Run the following command line to generate pseudo RGB images and fine-tune current model. You can prepare data according to the 10X and label data format in the Demo folder. Config file can be customized according to your needs. Please download the corresponding pre-trained model from [click here for downloading data and model](https://bmbl.bmi.osumc.edu/downloadFiles/data_and_model/data_and_model.zip) and put it in the specified folder.
```
wget https://bmbl.bmi.osumc.edu/downloadFiles/RESEPT/RESEPT.zip 
unzip RESEPT.zip
python training_pipeline.py -data ./Demo -config ./configs/config.py -out Demo_result -method scGNN  -transform logcpm -checkpoint Demo/checkpoint/S13_scGNN.pth
```

#### Command Line Arguments:
* -data 10X data h5 file, tissue positions list file and scalefactors json file folder path. [type:str]
* -config training config file path. [type:str]
*	-checkpoint specify path for pretrained model file. [type:str]
*	-out specify output root folder. [type:str]
*	-method specify embedding method in use: scGNN or spaGCN. [type:str]
*	-transform specify data pre-transform: log, logcpm or None. [type:str]

#### Expected Results
RESEPT stores the generative results in the following structure:
   ```
      Demo_result/
      |__RGB_images/
      |__RGB_images_label/
      work_dirs/
      |__Config/
            |__epoch_n.pth
   ```
*	-The folder 'RGB_images' stores generative RGB images from different embedding parameters. 
*	-The folder 'RGB_images_label' stores generative RGB images label from label file. 
*	-The file 'epoch_n.pth' is a final model after fine tune.
*	-This Demo takes 1 hour to generate all results on a machine with a 2080Ti GPU.

## Built With
 
* [opencv](https://opencv.org/) - The image processing library used
* [pytorch](https://pytorch.org/) - The deep learning backend used
* [scikit-learn](https://scikit-learn.org/stable/) - The machine learning library used
* [mmSegmentation](https://github.com/open-mmlab/mmsegmentation) - Used to train the deep learning based image segmentation model
 
## License
 
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
 
## Citation
if you use RESEPT software, please cite the following article:

https://www.biorxiv.org/content/10.1101/2021.07.08.451210v2
