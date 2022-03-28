# Neural Points
This repo is the implementation of the paper: Neural Points: Point Cloud Representation with Neural Fields for Arbitrary Upsampling (CVPR 2021).

- Paper address: [https://arxiv.org/abs/2112.04148](https://arxiv.org/abs/2112.04148)
- Project webpage: [https://wanquanf.github.io/NeuralPoints.html](https://wanquanf.github.io/NeuralPoints.html)


![avatar](./utils/Pipeline_v5.png)

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


## Citation
Please cite this paper with the following bibtex:

    @inproceedings{feng2022np,
        author    = {Wanquan Feng and Jin li and Hongrui Cai and Xiaonan Luo and Juyong Zhang},
        title     = {Neural Points: Point Cloud Representation with Neural Fields for Arbitrary Upsampling},
        booktitle = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition (CVPR)},
        year      = {2022}
    }

