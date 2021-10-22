# AutoMTL
This is the website for our paper "AutoMTL: A Programming Framework for Automated Multi-Task Learning", submitted to MLSys 2022. 
The arXiv version will be public at Mon, 25 Oct 2021.

### Abstract
Multi-task learning (MTL) jointly learns a set of tasks. It is a promising approach to reduce the training and inference time and storage costs while improving prediction accuracy and generalization performance for many computer vision tasks. However, a major barrier preventing the widespread adoption of MTL is the lack of systematic support for developing compact multi-task models given a set of tasks. In this paper, we aim to remove the barrier by developing the first programming framework AutoMTL that automates MTL model development. AutoMTL takes as inputs an arbitrary backbone convolutional neural network and a set of tasks to learn, then automatically produce a multi-task model that achieves high accuracy and has small memory footprint simultaneously. As a programming framework, AutoMTL could facilitate the development of MTL-enabled computer vision applications and even further improve task performance.

### Environment
PyTorch 1.6 or higher

### Datasets

## Code Description

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

## How to Use
[TODO: detailed explanations]
