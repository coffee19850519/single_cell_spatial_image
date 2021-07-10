# RESEPT: a computational framework for REconstructing and Segmenting Expression pseudo-image based on sPatially resolved Transcriptomics
 
A novel method to reconstruct a pseudo-image of spots using the sequencing data from spatially resolved transcriptomics to identify spatial context and functional zonation.
 
## Getting Started
 
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.
 
### Prerequisites
 
What things you need to install the software and how to install them
 
```
pip install umap-learn
pip install scikit-learn
pip install python-opencv
pip install mmsegmentation
```
 
## Running the tests
 
Explain how to run the automated tests for this system
 
### And coding style tests
 
Explain what these tests test and why
 
```
Give an example
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

## Generate pseudo-color images
```
python  pseudoimages_pipeline.py -matrix *.h5  -csv *.csv  -json *.json  -out * -method *  -gene *.txt
```
* -matrix  10X data h5 file path
* -csv tissue positions list file path
* -json scalefactors json file path
* -out output folder
* -method scGNN or spaGCN  [default:scGNN]
* -panel gene txt file path,one line is a panel gene. Default involved all genes. When specify gene list, involved sprcific genes. [optional][default:None]


## Segmentation to pseudo-color images
```
python  test_pipeline.py -matrix *.h5  -csv *.csv  -json *.json  -out * -method *  -gene *.txt
```
* -matrix  10X data h5 file path
* -csv tissue positions list file path
* -json scalefactors json file path
* -out output folder name [optional][default:output]
* -method scGNN or spaGCN  [default:scGNN]
* -gene txt file path,one line is a panel gene. Default involved all genes. When specify gene list, involved sprcific genes. [optional][default:None]


## Segmentation to optical images 
```
python  optical_segmentation_pipeline.py -matrix *.h5  -csv *.csv  -json *.json  -optical *.png  -out * -method * 
```
* -matrix  10X data h5 file path
* -csv tissue positions list file path
* -json scalefactors json file path
* -optical optical image path
* -out output folder name [optional][default:output]
* -method scGNN or spaGCN  [default:scGNN]


## Segmentation evaluation 
```
python  evaluation_pipeline.py py -matrix *.h5  -csv *.csv  -json *.json  -out *  -method *  -label *.csv
```
* -matrix  10X data h5 file path
* -csv tissue positions list file path
* -json scalefactors json file path
* -out output folder name [optional][default:output]
* -method scGNN or spaGCN  [default:scGNN]
* -label file path. One column is barcode and one column is corresponding label.

## Case study
```
python case_study_pipeline.py -matrix *.h5 -csv *.csv -json *.json -out * -gene *.txt  -method * -red_min * -red_max * -green_min *  -green_max * -blue_min * -blue_max *
```
* -matrix  10X data h5 file path
* -csv tissue positions list file path
* -json scalefactors json file path
* -out output folder name [optional][default:output]
* -gene txt file path,one line is a panel gene. Default involved all genes. When specify gene list, involved sprcific genes. [optional][default:None]
* -method scGNN or spaGCN  [default:scGNN]
* -red_min The lower limit of channel red [int]
* -red_max The upper limit of channel red [int]
* -green_min The lower limit of channel green [int]
* -green_max The upper limit of channel green [int]
* -blue_min The lower limit of channel blue [int]
* -blue_max The upper limit of channel blue [int]

## Training pipeline
```
python training_pipeline.py -data */ -config *.py -model *.pth -out * -gene *.txt -method * 
```
* -matrix  10X data h5 file , tissue positions list file and scalefactors json file path
* -config config file path
* -model resume file path
* -out output folder name [optional][default:output]
* -gene txt file path,one line is a panel gene. Default involved all genes. When specify gene list, involved sprcific genes. [optional][default:None]
* -method scGNN or spaGCN  [default:scGNN]

