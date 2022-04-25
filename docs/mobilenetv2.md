# Mobilenetv2

A [model](https://github.com/tonylins/pytorch-mobilenet-v2) that embedded with [Multi-Task-Learning-training](https://github.com/zhanglijun95/AutoMTL/blob/main/mtl_pytorch/mobilenetv2.py)

## Implementation 
* Replace every `Conv2d` and `BN2d` Layer in pytorch with `layer_node.Conv2d()`, or `layer_node.BN2d()`, with your interested task
and wrap `Sequential` using `layer_node.Sequential`