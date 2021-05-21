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

## Generate pseudo images 
python  panelgene_pipeline.py 2-5 2-8 -p Cartpt Mfge8 -r 0.5 -rl 0.2 0.7 0.05 -in original -out generate_pseudo_images

* 2-5 2-8... : sample list
* -p panel gene list
* -r specified resolution value,default:0.65
* -rl resolution range ,input first,last and step (alternative -r and -rl)
* -in original file folder,contains 10x folders with the sample name
* -out generate pseudo images folder

## Segmentation optical images 
python  segmentation_pipeline.py 2-5 2-8 -p Cartpt Mfge8 -r 0.5 -rl 0.2 0.7 0.05 -in original -out generate_pseudo_images

* 2-5 2-8... : sample list
* -p panel gene list
* -r specified resolution value,default:0.65
* -rl resolution range ,input first,last and step (alternative -r and -rl)
* -in original file folder,contains 10x folders with the sample name
* -out generate pseudo images folder
