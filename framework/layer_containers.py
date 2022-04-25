import torch
import torch.nn as nn

class EltwiseOp(nn.Module):
    def __init__(self, operation):
        # Funtion: Perform element-wise operations
        #          Correponding to the EltwiseOp Enum in pytorch.proto
        #          0: Product; 1: Add; 2: Max
        super(EltwiseOp, self).__init__()
        self.op = operation
        
    def forward(self, x, y):
        if self.op == 0:
            return torch.prod(x, y)
        elif self.op == 1:
            return torch.add(x, y)
        elif self.op == 2:
            return torch.max(x, y)
        
    def extra_repr(self):
        return 'op=' + str(self.op)
    

class AbstractPool(nn.Module):
    def __init__(self, pool_method, global_pooling, kernel_size, stride, padding, ceil_mode):
        # Funtion: Perform different pooling
        super(AbstractPool, self).__init__()
        #          Correponding to the PoolMethod Enum in pytorch.proto
        #          0: Max; 1: Avg
        self.pool_method = pool_method
        self.global_pooling = global_pooling
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if ceil_mode == 0:
            self.ceil_mode = True
        else:
            self.ceil_mode = False
        
        self.set_pooling_op()
        
    def set_pooling_op(self):
        if self.global_pooling:
            if self.pool_method == 0:
                self.pool_op = nn.AdaptiveMaxPool2d((1,1))
            elif self.pool_method == 1:
                self.pool_op = nn.AdaptiveAvgPool2d((1,1))
        else:
            if self.pool_method == 0:
                self.pool_op = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, ceil_mode=self.ceil_mode)
            elif self.pool_method == 1:
                self.pool_op = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, ceil_mode=self.ceil_mode)
            
    def forward(self, x):
        return self.pool_op(x)
    
#     def extra_repr(self):
#         if self.global_pooling:
#             return 'global_pooling=' + str(self.global_pooling) + ', pool_method=' + str(self.pool_method)
#         else:
#             return 'pool_method=' + str(self.pool_method) + ', kernel_size=' + str(self.kernel_size) + \
#                     ', stride=' + str(self.stride) + ', padding=' + str(self.padding) + ', ceil_mode=' + str(self.ceil_mode)


class LazyLayer(nn.Module):
    def __init__(self):
        super(LazyLayer, self).__init__()
    
    def forward(self, x):
        return x