# Code for CVPR2022 Submission
The code is for the paper (submitted to CVPR 2022): Neural Points: Point Cloud Representation with Neural Fields.

## Prerequisite Installation
The code has been tested with Python3.8, PyTorch 1.6 and Cuda 10.2:

    conda create --name NePs
    
    conda activate NePs
    
    conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.2 -c pytorch
    
    conda install -c conda-forge igl
    
Before running the code, you need to build the cuda&C++ extensions of Pytorch:

    cd [ProjectPath]/model/model_for_supp/pointnet2
    
    python setup.py install

    
## How to use the code: 
Download our dataset: [dataset](https://pan.baidu.com/s/1BLFobnIkuLqrXsdAAVqA0g), (extracting code: qiqq). Put the 'Sketchfab2' folder into: [ProjectPath]/data.

Firstly, you need to change the working directory: 

    cd [ProjectPath]/model/conpu_v6

To obtain the testing results of the testing set, run:

    python train_script101_test.py

To train our network, run:

    python train_script101.py

    

