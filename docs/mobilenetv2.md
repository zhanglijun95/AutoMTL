# Mobilenetv2

A [model](https://github.com/tonylins/pytorch-mobilenet-v2) that embedded with [Multi-Task-Learning-training](https://github.com/zhanglijun95/AutoMTL/blob/main/APIs/mobilenetv2.py)

## Implementation 
* Replace every `Conv2d` and `BN2d` Layer in pytorch with `layer_node.Conv2dNode`, or `layer_node.BN2dNode`, with your interested task
and wrap `Sequential` using `layer_node.Sequential`
* Inherited from `mtl_model.mtl_model` and call `self.compute_depth()` at the end of `__init__()`.