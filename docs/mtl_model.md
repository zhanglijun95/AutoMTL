# [MTL Model](https://github.com/zhanglijun95/AutoMTL/blob/main/APIs/mtl_model.py)
## mtl_model
```python
mtl_model.mtl_model()
```

Parent class for customized multi-task model.

### function
* `compute_depth()` should be called at the end of the `__init__()` for the child model class inherited from `mtl_model`. See example in `[mobilenetv2.py](https://github.com/zhanglijun95/AutoMTL/blob/main/APIs/mobilenetv2.py)`.
* `policy_reg()` will be called automatically when running `alter_train_with_reg()` of `Trainer`.
* **Note: If choose to not inherit from `mtl_model`, please run `alter_train()` of `Trainer` at the second alter_train stage.**