# AutoMTL: A Programming Framework for Automated Multi-Task Learning 
This is the website for our paper "AutoMTL: A Programming Framework for Automated Multi-Task Learning", submitted to ICML 2022. 
The arXiv version can be found [here](https://arxiv.org/pdf/2110.13076.pdf).

### Abstract
Multi-task learning (MTL) jointly learns a set of tasks. It is a promising approach to reduce the training and inference time and storage costs while improving prediction accuracy and generalization performance for many computer vision tasks. However, a major barrier preventing the widespread adoption of MTL is the lack of systematic support for developing compact multi-task models given a set of tasks. In this paper, we aim to remove the barrier by developing the first programming framework AutoMTL that automates MTL model development. AutoMTL takes as inputs an arbitrary backbone convolutional neural network and a set of tasks to learn, then automatically produce a multi-task model that achieves high accuracy and has small memory footprint simultaneously. As a programming framework, AutoMTL could facilitate the development of MTL-enabled computer vision applications and even further improve task performance.

![overview](https://github.com/zhanglijun95/AutoMTL/blob/main/assets/overview.jpg)

### Cite
Welcome to cite our work if you find it is helpful to your research.
```
@misc{zhang2021automtl,
      title={AutoMTL: A Programming Framework for Automated Multi-Task Learning}, 
      author={Lijun Zhang and Xiao Liu and Hui Guan},
      year={2021},
      eprint={2110.13076},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# Description
### Environment
```bash
conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch # Or higher
conda install protobuf
pip install opencv-python
pip install scikit-learn
```

### Datasets
We conducted experiments on three popular datasets in multi-task learning (MTL), **CityScapes** [[1]](#1), **NYUv2** [[2]](#2), and **Tiny-Taskonomy** [[3]](#3). You can download the them [here](https://drive.google.com/drive/folders/1KX9chooxefvrZACtFR441ShdkbS3F2lt?usp=sharing). For Tiny-Taskonomy, you will need to contact the authors directly. See their [official website](http://taskonomy.stanford.edu/).

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
  *  Dataloaders ```*_dataloader.py```: 
  For each dataset, we offer a corresponding PyTorch dataloader with a specific _task_ variable.
  *  Heads ```pixel2pixel.py```: 
  The **ASPP** head [[4]](#4) is implemented for the pixel-to-pixel vision tasks.
  *  Metrics ```pixel2pixel_loss/metrics.py```: 
  For each task, it has its own criterion and metric.

* AutoMTL
  * Multi-Task Model Generator ```mtl_model.py```:
  Transfer the given backbone model in the format of _prototxt_, and the task-specific model head dictionary to a multi-task supermodel.
  * Trainer Tools ```trainer.py```:
  Meterialize a three-stage training pipeline to search out a good multi-task model for the given tasks.
  ![pipeline](https://github.com/zhanglijun95/AutoMTL/blob/main/assets/pipeline.jpg)
  
* Others
  *  Input Backbone ```*.prototxt```:
  Typical vision backbone models including **Deeplab-ResNet34** [[4]](#4), **MobileNetV2**, and **MNasNet**.
  *  Transfer to Prototxt ```pytorch_to_caffe.py```:
  If you define your own customized backbone model in PyTorch API, we also provide a tool to convert it to a prototxt file.

# How to Use
**Note**: Please refer to ```Example.ipynb``` for more details. 

## Set up Data
Each task will have its own **dataloader** for both training and validation, **task-specific criterion (loss), evaluation metric, and model head**. Here we take CityScapes as an example. 
``` bash
tasks = ['segment_semantic', 'depth_zbuffer']
task_cls_num = {'segment_semantic': 19, 'depth_zbuffer': 1} # the number of classes in each task
```

You can also define your own dataloader, criterion, and evaluation metrics. Please refer to files in ```data/``` to make sure your customized classes have the same output format as ours to fit for our framework.

### dataloader dictionary
``` bash
trainDataloaderDict = {}
valDataloaderDict = {}
for task in tasks:
    dataset = CityScapes(dataroot, 'train', task, crop_h=224, crop_w=224)
    trainDataloaderDict[task] = DataLoader(dataset, <batch_size>, shuffle=True)

    dataset = CityScapes(dataroot, 'test', task)
    valDataloaderDict[task] = DataLoader(dataset, <batch_size>, shuffle=True)
```

### criterion dictionary
``` bash
criterionDict = {}
for task in tasks:
    criterionDict[task] = CityScapesCriterions(task)
```

### evaluation metric dictionary
``` bash
metricDict = {}
for task in tasks:
    metricDict[task] = CityScapesMetrics(task)
```

### task-specific heads dictionary
``` bash
headsDict = nn.ModuleDict() # must be nn.ModuleDict() instead of python dictionary
for task in tasks:
    headsDict[task] = ASPPHeadNode(<feature_dim>, task_cls_num[task])
```

## Construct Multi-Task Supermodel
``` bash
prototxt = 'models/deeplab_resnet34_adashare.prototxt' # can be any CNN model
mtlmodel = MTLModel(prototxt, headsDict)
```
**Note**: We currently support Conv2d, BatchNorm2d, Linear, ReLU, Droupout, MaxPool2d and AvgPool2d (including global pooling), elementwise operators (inclduing production, add, and max).

## 3-Stage Training
### define the trainer
``` bash
trainer = Trainer(mtlmodel, trainDataloaderDict, valDataloaderDict, criterionDict, metricDict)
```
### pre-train phase
``` bash
trainer.pre_train(iters=<total_iter>, lr=<init_lr>, savePath=<save_path>)
```

### policy-train phase
``` bash
loss_lambda = {'segment_semantic': 1, 'depth_zbuffer': 1, 'policy':0.0005} # the weights for each task and the policy regularization term from the paper
trainer.alter_train_with_reg(iters=<total_iter>, policy_network_iters=<alter_iters>, policy_lr=<policy_lr>, network_lr=<network_lr>, 
                             loss_lambda=loss_lambda, savePath=<save_path>)
```
**Note**: When training the policy and the model weights together, we alternatively train them for specified iters in ```policy_network_iters```.

### sample policy from trained policy distribution
``` bash
sample_policy_dict = OrderedDict()
for task in tasks:
    for name, policy in zip(name_list[task], policy_list[task]):
        distribution = softmax(policy, axis=-1)
        distribution /= sum(distribution)
        choice = np.random.choice((0,1,2), p=distribution)
        if choice == 0:
            sample_policy_dict[name] = torch.tensor([1.0,0.0,0.0]).cuda()
        elif choice == 1:
            sample_policy_dict[name] = torch.tensor([0.0,1.0,0.0]).cuda()
        elif choice == 2:
            sample_policy_dict[name] = torch.tensor([0.0,0.0,1.0]).cuda()
```
**Note**: The policy-train stage only obtains a good policy distribution. Before conducting post-train, we should sample a certain policy from the distribution.

### post-train phase
``` bash
trainer.post_train(ters=<total_iter>, lr=<init_lr>, 
                   loss_lambda=loss_lambda, savePath=<save_path>, reload=<sampled_policy>)
```

## Validation Results in the Paper
You can download fully-trained models for each dataset [here](https://drive.google.com/drive/folders/16dYXhyeZt2jgMyt3B4uDtcUhENIC1TUZ?usp=sharing).
``` bash
mtlmodel.load_state_dict(torch.load(<model_name>))
trainer.validate('mtl', hard=True)
```
**Note**: The "hard" must be set to True when conducting inference since we don't want to have soft policy this time.

## Inference from Trained Model
``` bash 
mtlmodel.load_state_dict(torch.load(<model_name>))
output = mtlmodel(x, task=<task_name>, hard=True)
```

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

<a id="4">[4]</a> 
Chen, Liang-Chieh and Papandreou, George and Kokkinos, Iasonas and Murphy, Kevin and Yuille, Alan L. 
Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. 
PAMI, 834-848, 2017.
