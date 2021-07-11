# RESEPT: a computational framework for REconstructing and Segmenting Expression pseudo-image based on sPatially resolved Transcriptomics
 
A novel method to reconstruct a pseudo-image of spots using the sequencing data from spatially resolved transcriptomics to identify spatial context and functional zonation.
 
## Getting Started
 
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.
 
### Prerequisites
 
What things you need to install the software and how to install them
 
```
pip install umap-learn
pip install scikit-learn
pip install scikit-image
pip install mmcv-full
pip install mmsegmentation
pip install scanpy
pip install opencv-python
pip install anndata
```
 
## Running the tests
 
### Generate pseudo RGB images
Program **pseudo-images_pipeline.py** is used to generate pseudo RGB images. It can also output the corresponding case study image according to the input panel gene txt file. Images generated in this step store in the pseudo_images folder under specified output folder.  Original 10X data files of 16 samples can be found on Baidu Cloud Disk[].

In **pseudo-images_pipeline.py** ,these parameters are used:

**Required**
* **-matrix** 10X data h5 file path.
* **-csv** tissue positions list file path.
* **-json** scalefactors json file path.
* **-out** output folder.
* **-method** generate embedding method:scGNN or spaGCN  [default:scGNN]

**Optional**
* **-gene** gene txt file path,one line is a panel gene. Default involved all genes. When specify gene list, involved sprcific genes. [optional][default:None]
* **-pca** pca option when generating  case study image. [optional][default:True]
* **-transform** data preproccessing method: log or logcpm or None.[default:None]

```
python  pseudo-images_pipeline.py -matrix *.h5  -csv *.csv  -json *.json  -out * -method *  -gene *.txt -pca * -transform *
```

### Segmentation to pseudo RGB images
Program **test_pipeline.py** is used to use the existing checkpoint to segmentation the generated pseudo RGB images. The top5 results generated after MI ranking are presented to the user. The category maps, visualizations and MI values corresponding to top5 are stored in the segmentation_test folder under specified output folder. Original 10X data files of 16 samples and checkpoint file can be found on Baidu Cloud Disk[].

In **test_pipeline.py** ,these parameters are used:

**Required**
* **-matrix** 10X data h5 file path.
* **-csv** tissue positions list file path.
* **-json** scalefactors json file path.
* **-out** output folder.
* **-method** generate embedding method:scGNN or spaGCN  [default:scGNN]
* **-checkpoint** checkpoint path

**Optional**
* **-gene** gene txt file path,one line is a panel gene. Default involved all genes. When specify gene list, involved sprcific genes. [optional][default:None]
* **-pca** pca option when generating  case study image. [optional][default:True]
* **-transform** data preproccessing method: log or logcpm or None.[default:None]
```
python  test_pipeline.py -matrix *.h5  -csv *.csv  -json *.json  -out * -method *  -gene *.txt -pca * -transform * -checkpoint *
```

### Segmentation to optical images 
Program **optical_segmentation_pipeline.py** is used to use the existing checkpoint to segmentation the optical images and generated pseudo RGB images. The category maps and visualizations are stored in the  optical_segmentation folder under specified output folder. The Original 10X data file and and checkpoint file in the manuscript can be found on Baidu Cloud Disk[].

**Required**
* **-matrix** 10X data h5 file path.
* **-csv** tissue positions list file path.
* **-json** scalefactors json file path.
* **-out** output folder.
* **-method** generate embedding method:scGNN or spaGCN  [default:scGNN]
* **-checkpoint** checkpoint path

**Optional**
* **-pca** pca option when generating  case study image. [optional][default:True]
* **-transform** data preproccessing method: log or logcpm or None.[default:None]

```
python  optical_segmentation_pipeline.py -matrix *.h5  -csv *.csv  -json *.json  -optical *.png  -out * -method * -pca * -transform * -checkpoint *
```

### Evaluation of segmentation results 
Program **evaluation_pipeline.py** is used to evaluate the segmentation results. User submits 10X and the corresponding label file to generate the pseudo RGB images, the visualizations of the top5 after MI ranking and the corresponding values of the evaluation index such as ARI. These output files are stored in the segmentation_evaluation folder under specified output folder. Original 10X data files and label files of 16 samples and checkpoint file can be found on Baidu Cloud Disk[].

In **evaluation_pipeline.py** ,these parameters are used:

**Required**
* **-matrix** 10X data h5 file path.
* **-csv** tissue positions list file path.
* **-json** scalefactors json file path.
* **-out** output folder.
* **-method** generate embedding method:scGNN or spaGCN  [default:scGNN]
* **-label** csv file path. One column is barcode and one column is corresponding label.
* **-checkpoint** checkpoint path

**Optional**
* **-pca** pca option when generating  case study image. [optional][default:True]
* **-transform** data preproccessing method: log or logcpm or None.[default:None]

```
python  evaluation_pipeline.py  -matrix *.h5  -csv *.csv  -json *.json  -out *  -method * -pca * -transform * -label *.csv -checkpoint *
```

### Case study
Program **case_study_pipeline.py** is used to generate pseudo RGB images and use specific RGB parameters to obtain a filtered image of a specific area. The pseudo RGB images and filtered images are stored in the case_study folder under specified output folder. Original 10X data files of 16 samples can be found on Baidu Cloud Disk[].

In **case_study_pipeline.py** ,these parameters are used:

**Required**
* **-matrix** 10X data h5 file path
* **-csv** tissue positions list file path
* **-json** scalefactors json file path
* **-out** output folder name [optional][default:output]
* **-gene** txt file path,one line is a panel gene. Default involved all genes. When specify gene list, involved sprcific genes. [optional][default:None]
* **-method** generate embedding method:scGNN or spaGCN  [default:scGNN]
* **-red_min** The lower limit of channel red [int]
* **-red_max** The upper limit of channel red [int]
* **-green_min** The lower limit of channel green [int]
* **-green_max** The upper limit of channel green [int]
* **-blue_min** The lower limit of channel blue [int]
* **-blue_max** The upper limit of channel blue [int]

**Optional**
* **-pca** pca option when generating  case study image. [optional][default:True]
* **-transform** data preproccessing method: log or logcpm or None.[default:None]

```
python case_study_pipeline.py -matrix *.h5 -csv *.csv -json *.json -out * -gene *.txt  -method * -pca * -transform * -red_min * -red_max * -green_min *  -green_max * -blue_min * -blue_max *
```

### Training pipeline
Program **training_pipeline.py** is used to generate pseudo RGB images and fine-tune current model. Config file can be customized according to your needs. Cheakpoint files can be found on Baidu Cloud Disk[]. The new cheakpoint is stored in the work_dir folder.

In **training_pipeline.py** ,these parameters are used:

**Required**
* **-data** 10X data h5 file, tissue positions list file and scalefactors json file folder path.
* **-config** training config file path.
* **-model** resume model path.[default:None]
* **-gene** txt file path,one line is a panel gene. Default involved all genes. When specify gene list, involved sprcific genes. [optional][default:None]
* **-method** generate embedding method:scGNN or spaGCN  [default:scGNN]
* **-pca** pca option when generating  case study image. [optional][default:True]
* **-transform** data preproccessing method: log or logcpm or None.[default:None]

**Optional**
* **-pca** pca option when generating  case study image. [optional][default:True]
* **-transform** data preproccessing method: log or logcpm or None.[default:None]

```
python training_pipeline.py -data * -config * -model * -gene * -method * -pca * -transform *
```

 
## Built With
 
* [opencv](https://opencv.org/) - The image processing library used
* [scikit-learn](https://scikit-learn.org/stable/) - The machine learning library used
* [mmSegmentation](https://github.com/open-mmlab/mmsegmentation) - Used to train the deep learning based image segmentation model
 
## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.
  
## License
 
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
 
## Acknowledgments
 
* Hat tip to anyone whose code was used
* Inspiration
* etc
