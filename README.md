# AutoMTL: A Programming Framework for Automated Multi-Task Learning 
This is the website for our paper "AutoMTL: A Programming Framework for Automated Multi-Task Learning", submitted to MLSys 2022. 
The arXiv version will be public at Tue, 26 Oct 2021.

### Abstract
Multi-task learning (MTL) jointly learns a set of tasks. It is a promising approach to reduce the training and inference time and storage costs while improving prediction accuracy and generalization performance for many computer vision tasks. However, a major barrier preventing the widespread adoption of MTL is the lack of systematic support for developing compact multi-task models given a set of tasks. In this paper, we aim to remove the barrier by developing the first programming framework AutoMTL that automates MTL model development. AutoMTL takes as inputs an arbitrary backbone convolutional neural network and a set of tasks to learn, then automatically produce a multi-task model that achieves high accuracy and has small memory footprint simultaneously. As a programming framework, AutoMTL could facilitate the development of MTL-enabled computer vision applications and even further improve task performance.

![overview](https://github.com/zhanglijun95/AutoMTL/blob/main/assets/overview.jpg)

### Cite
Welcome to cite our work if you find it is helpful to your research.
[TODO: cite info]

# Description
### Environment
```bash
conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch # Or higher
conda install protobuf
pip install opencv-python
pip install scikit-learn
```

### Datasets
We conducted experiments on three popular datasets in multi-task learning (MTL), **CityScapes** [[1]](#1), **NYUv2** [[2]](#2), and **Tiny-Taskonomy** [[3]](#3). You can download the them [here](https://drive.google.com/file/d/1YyJ-smgkagwpSU5F1oBH8UkN06-TW3W7/view?usp=sharing). For Tiny-Taskonomy, you will need to contact the authors directly. See their [official website](http://taskonomy.stanford.edu/).

### File Structure
```bash
├── data
│   ├── dataloader
│   │   ├── *_dataloader.py
│   ├── heads
│   │   ├── pixel2pixel.py
│   ├── metrics
│   │   ├── pixel2pixel_loss/metrics.py
├── framework
│   ├── layer_containers.py
│   ├── base_node.py
│   ├── layer_node.py
│   ├── mtl_model.py
│   ├── trainer.py
├── models
│   ├── *.prototxt
├── utils
└── └── pytorch_to_caffe.py
```

### Code Description
Our code can be divided into three parts: code for data, code of AutoMTL, and others

* For Data
  *  Dataloaders
  *  Heads
  *  Metrics

* AutoMTL
  * Multi-Task Model Generator
  * Trainer Tools
  ![pipeline](https://github.com/zhanglijun95/AutoMTL/blob/main/assets/pipeline.jpg)
  
* Others
  *  Input Backbone
  *  Transfer to Prototxt

# How to Use
[TODO: detailed explanations]

# References
<a id="1">[1]</a> 
Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler, Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt. 
The cityscapes dataset for semantic urban scene understanding. 
CVPR, 3213-3223, 2016.

<a id="2">[2]</a> 
Silberman, Nathan and Hoiem, Derek and Kohli, Pushmeet and Fergus, Rob. 
Indoor segmentation and support inference from rgbd images. 
ECCV, 746-760, 2012.

<a id="3">[3]</a> 
Zamir, Amir R and Sax, Alexander and Shen, William and Guibas, Leonidas J and Malik, Jitendra and Savarese, Silvio. 
Taskonomy: Disentangling task transfer learning. 
CVPR, 3712-3722, 2018.
