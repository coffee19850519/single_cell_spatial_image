[docs-image]: https://readthedocs.org/projects/pytorch-geometric/badge/?version=latest
[docs-url]: https://resept-last.readthedocs.io/en/latest/Case%20study.html

# Define and visualize pathological architectures of human tissues from spatially resolved transcriptomics using deep learning
**[Paper](https://www.biorxiv.org/content/10.1101/2021.07.08.451210v1)** 
  
```RESEPT``` is a deep-learning framework for characterizing and visualizing tissue architecture from spatially resolved transcriptomics. 

Given inputs as gene expression or RNA velocity, ```RESEPT``` learns a three-dimensional embedding with a spatial retained graph neural network from the spatial transcriptomics. The embedding is then visualized by mapping as color channels in an RGB image and segmented with a supervised convolutional neural network model to infer the tissue architecture accurately. 


<p align="center">
  <img height="300" width="700" src="https://github.com/yuyang-0825/image/blob/main/figure1.png" />
</p>

 
## System Requirements

### Hardware Requirements
 
``` RESEPT ``` requires a standard computer with enough RAM to support the in-memory operations. A GPU with enough VRAM or multi-core CPU is recommended for acceleration.

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

## Data preparation

### 10x Visium data
 * gene expression file: A HDF5 file stores raw gene expression data.  
 * tissue_positions_list file: A csv file stores meta of spot including their connectivity and spatial coordinates.
 * scalefactors_json file: A json file stores the scaling factors converting spots to different resolutions.

### Annotation file

An annotation file recording spot barcodes and their corresponding annotations. It is required in the functions of evaluating predictive tissue architectures and customizing segmentation model. The file should be named as: [sample_name]_annotation.csv. [example](https://bmbl.bmi.osumc.edu/downloadFiles/data_and_model/data_and_model.zip) (/ocean/projects/ccr180012p/shared/Demo/S13/S13_annotation.csv)

### Segmentation model file

A trained segmentation model file in pth format. It is required to predict tissue architecture on generative visuals.

### Data structure

The data schema to run our code is as following:
```
[sample_name]/
 |__spatial/
 |    |__tissue_positions_list file
 |    |__scalefactors_json file
 |__gene expression file
 |__annotation file: [sample_name]_annotation.csv (optional)

model/ (optional)
 |__segmentation model file 
   ```

## Demo

### Visualize tissue architecture 
Run the following command line to generate visuals of gene expression from different embedding parameters. For demonstration, please download the example data from [here](https://bmbl.bmi.osumc.edu/downloadFiles/data_and_model/data_and_model.zip)(/ocean/projects/ccr180012p/shared/Demo/S13) and put the unzip folder 'S13' to source code folder.
```
wget https://bmbl.bmi.osumc.edu/downloadFiles/RESEPT/RESEPT.zip 
unzip RESEPT.zip
python RGB_images_pipeline.py -expression S13/S13_filtered_feature_bc_matrix.h5  -meta S13/spatial/tissue_positions_list.csv  -scaler S13/spatial/scalefactors_json.json -output Demo_result  -embedding scGNN  -transform logcpm 
```

#### Command Line Arguments:
*	-expression file path for raw gene expression data. [type:str]
*	-meta file path for spatial meta recording tissue positions. [type:str]
*	-scaler file path for scale factors. [type:str]
*	-output output root folder. [type:str]
*	-embedding embedding method in use: scGNN or spaGCN. [type:str]
*	-transform data pre-transform method: log, logcpm or None. [type:str]

#### Results
RESEPT stores the generative results in the following structure:
   ```
      Demo_result/
      |__RGB_images/
   ```
*	-The folder 'RGB_images' stores generative visuals of tissue architectures from different embedding parameters. 

### Evaluate predictive tissue architecture with annotations
Run the following command line to generate visuals of gene expression from different embedding parameters, segmentation maps with top5 Moran's I and their evaluation metrics. For demonstration, please download the example data and pre-trained model from [here](https://bmbl.bmi.osumc.edu/downloadFiles/data_and_model/data_and_model.zip) and put the unzip folder 'Demo' to source code folder.
```
wget https://bmbl.bmi.osumc.edu/downloadFiles/RESEPT/RESEPT.zip 
unzip RESEPT.zip
python evaluation_pipeline.py -expression Demo/S13/S13_filtered_feature_bc_matrix.h5  -meta Demo/S13/spatial/tissue_positions_list.csv  -scaler Demo/S13/spatial/scalefactors_json.json -output Demo_result  -embedding scGNN  -transform logcpm -label Demo/S13/S13_annotation.csv -model Demo/model/S13_scGNN.pth
```

#### Command Line Arguments:
*	-expression file path for raw gene expression data. [type:str]
*	-meta file path for spatial meta recording tissue positions. [type:str]
*	-scaler file path for scale factors. [type:str]
*	-label file path for labels recording cell barcodes and their annotations for calculating evaluation metrics. [type:str]
*	-model file path for pretrained model. [type:str]
*	-output output root folder. [type:str]
*	-embedding embedding method in use: scGNN or spaGCN. [type:str]
*	-transform data pre-transform method: log, logcpm or None. [type:str]

#### Results
RESEPT stores the generative results in the following structure:
   ```
      Demo_result/
      |__RGB_images/
      |__segmentation_evaluation/
            |__segmentation_map/
            |__top5_evaluation.csv
   ```
*	-The folder 'RGB_images' stores generative visuals of tissue architectures from different embedding parameters. 
*	-The folder 'segmentation_map' stores visuals of predictive tissue architectures with top5 Moran's I. 
*	-The file 'top5_evaluation.csv' records various evaluation metrics corresponding to the predictions.
*	-This Demo takes 30-35 mins to generate all results on a machine with a multi-core CPU.

### Predict tissue architecture without annotation
Run the following command line to generate visuals of gene expression from different embedding parameters and predict tissue architectures with top5 Moran's I. For demonstration, please download the example data and pre-trained model from [here](https://bmbl.bmi.osumc.edu/downloadFiles/data_and_model/data_and_model.zip) and put the unzip folder 'Demo' to source code folder.
```
wget https://bmbl.bmi.osumc.edu/downloadFiles/RESEPT/RESEPT.zip 
unzip RESEPT.zip
python test_pipeline.py -expression Demo/S13/S13_filtered_feature_bc_matrix.h5  -meta Demo/S13/spatial/tissue_positions_list.csv  -scaler Demo/S13/spatial/scalefactors_json.json -output Demo_result  -embedding scGNN  -transform logcpm -model Demo/model/S13_scGNN.pth
```

#### Command Line Arguments:
*	-expression file path for raw gene expression data. [type:str]
*	-meta file path for spatial meta file recording tissue positions. [type:str]
*	-scaler file path for scale factors. [type:str]
*	-model file path for pretrained model. [type:str]
*	-output output root folder. [type:str]
*	-embedding embedding method in use: scGNN or spaGCN. [type:str]
*	-transform data pre-transform method: log, logcpm or None. [type:str]

#### Results
RESEPT stores the generative results in the following structure:
   ```
      Demo_result/
      |__RGB_images/
      |__segmentation_test/
            |__segmentation_map/
            |__top5_MI_value.csv
   ```
*	-The folder 'RGB_images' stores generative visuals of tissue architectures from different embedding parameters. 
*	-The folder 'segmentation_map' stores visuals of predictive tissue architectures with top5 Moran's I. 
*	-The file 'top5_MI_value.csv' records Moran's I value corresponding to the predictions.
*	-This Demo takes 30-35 mins to generate all results on a machine with a multi-core CPU.


### Customize segmentation model 
RESEPT supports fine-tuning our segmentation model by using your own 10x data. Organize all 10x data and their labels according to our predefined data schema and download our pre-trained model from [here](https://bmbl.bmi.osumc.edu/downloadFiles/data_and_model/data_and_model.zip)(/ocean/projects/ccr180012p/shared/Demo/model_S13) as a training start point. The 10x data of each sample should be located in a separate sub-folder under the 'data folder'. For demonstration, download the example training data from [here](https://bmbl.bmi.osumc.edu/downloadFiles/data_and_model/data_and_model.zip)(/ocean/projects/ccr180012p/shared/Demo/training_data_folder), and then run the following command line to get the visuals of your own data and the customized model.  
```
wget https://bmbl.bmi.osumc.edu/downloadFiles/RESEPT/RESEPT.zip 
unzip RESEPT.zip
python training_pipeline.py -data_folder Demo -output Demo_result -embedding scGNN  -transform logcpm -model Demo/model/S13_scGNN.pth
```

#### Command Line Arguments:
* -data_folder a folder provides all training samples. The data including label file of each sample should follow our predefined schema in a sub-folder under this folder. [type:str]
*	-model file path for pretrained model file. [type:str]
*	-output output root folder. [type:str]
*	-embedding embedding method in use: scGNN or spaGCN. [type:str]
*	-transform data pre-transform method: log, logcpm or None. [type:str]

#### Results
RESEPT stores the generative results in the following structure:
   ```
      Demo_result/
      |__RGB_images/
      |__RGB_images_label/
      work_dirs/
      |__config/
            |__epoch_n.pth
   ```
*	-The folder 'RGB_images' stores generative visuals of tissue architectures of all input 10x data from different embedding parameters. 
*	-The folder 'RGB_images_label' stores their labeled category maps according to input label files. 
*	-The file 'epoch_n.pth' is the customized model.
*	-This Demo takes about 3 hours to generate the model on a machine with a 2080Ti GPU.

### Segment histological images
Run the following command line to generate visuals of gene expression from different embedding parameters, predict tissue architectures with top5 Moran's I and segment histological images according to tissue architectures. For demonstration, please download the example data from here (/ocean/projects/ccr180012p/shared/Demo/cancer)and pre-trained model from here (/ocean/projects/ccr180012p/shared/Demo/model_cancer) and put the unzip folder 'cancer ' and ‘model_cancer’to source code folder.
```
wget https://bmbl.bmi.osumc.edu/downloadFiles/RESEPT/RESEPT.zip 
unzip RESEPT.zip
python histological_segmentation_pipeline.py -expression ./cancer/Parent_Visium_Human_Glioblas_filtered_feature_bc_matrix.h5 -meta ./cancer/spatial/tissue_positions_list.csv -scaler ./cancer/spatial/scalefactors_json.json -optical ./cancer/Parent_Visium_Human_Glioblast.tif -output Demo_result -model ./model_cancer/cancer_model.pth -embedding spaGCN -transform logcpm
```

#### Command Line Arguments:
*	-expression file path for raw gene expression data. [type:str]
*	-meta file path for spatial meta file recording tissue positions. [type:str]
*	-scaler file path for scale factors. [type:str]
*	-model file path for pretrained model. [type:str]
*   -optical file path for an optical image.
*	-output output root folder. [type:str]
*	-embedding embedding method in use: scGNN or spaGCN. [type:str]
*  	-transform data pre-transform method: log, logcpm or None. [type:str]

#### Results
RESEPT stores the generative results in the following structure:
   ```
      Demo_result/
      |__RGB_images/
      |__segmentation_test/
            |__segmentation_map/
            |__top5_MI_value.csv
      |__optical_segmentation/
            |__category_n.png
   ```
*	-The folder 'RGB_images' stores generative visuals of tissue architectures from different embedding parameters. 
*	-The folder 'segmentation_map' stores visuals of predictive tissue architectures with top5 Moran's I. 
*	-The file 'top5_MI_value.csv' records Moran's I value corresponding to the predictions.
*	--The file 'category_n.png ' refers to the histological image segmentation results.
*	-This Demo takes 30-35 mins to generate all results on a machine with a multi-core CPU.


## Built With
 
* [opencv](https://opencv.org/) - The image processing library used
* [pytorch](https://pytorch.org/) - The deep learning backend used
* [scikit-learn](https://scikit-learn.org/stable/) - The machine learning library used
* [mmSegmentation](https://github.com/open-mmlab/mmsegmentation) - Used to train the deep learning based image segmentation model
 
## License
 
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
 
## Citation
if you use ```RESEPT```, please cite [our paper](https://www.biorxiv.org/content/10.1101/2021.07.08.451210v1):
```
@article {Chang2021.07.08.451210,
	author = {Chang, Yuzhou and He, Fei and Wang, Juexin and Chen, Shuo and Li, Jingyi and Liu, Jixin and Yu, Yang and Su, Li and Ma, Anjun and Allen, Carter and Lin, Yu and Sun, Shaoli and Liu, Bingqiang and Otero, Jose and Chung, Dongjun and Fu, Hongjun and Li, Zihai and Xu, Dong and Ma, Qin},
	title = {Define and visualize pathological architectures of human tissues from spatially resolved transcriptomics using deep learning},
	elocation-id = {2021.07.08.451210},
	year = {2021},
	doi = {10.1101/2021.07.08.451210},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2021/07/16/2021.07.08.451210},
	eprint = {https://www.biorxiv.org/content/early/2021/07/16/2021.07.08.451210.full.pdf},
	journal = {bioRxiv}
}
```
