# [layer_node](https://github.com/suikac/AutoMTL/blob/main/mtl_pytorch/layer_node.py)
## Conv2dNode
```python
layer_node.Conv2dNode(in_channels, out_channels, kernel_size, 
                    stride, padding, paddingmode, dilation, bias, 
                    groups, task_list)
```


Conv2d embedded with task. 

### argument
* arguments except `task_list` are the same as [Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
* `task_list` should be a list of tasks that you are interested in trainning

## BN2dNode
```python
layer_node.BN2dNode(num_features: int,
                    eps: float, 
                    momentum: float, 
                    affine: bool, 
                    track_running_stats: bool, 
                    task_list: list)
```

BN2d embedded with task

### argument
* arguments except `task_list` are the same as [BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
* `task_list` should be a list of tasks that you are interested in trainning


## Sequential 
```python
layer_node.Sequential(seq)
```

A useful wrapper class for applying special forwarding in nn.Sequential

### Argument
* `seq`: should be `nn.Sequential` type of module