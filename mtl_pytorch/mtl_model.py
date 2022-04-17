import warnings

import torch.nn as nn
import torch
from mtl_pytorch.layer_node import Conv2dNode, BN2dNode
from copy import deepcopy

class mtl_model(nn.Module):

    def __init__(self, model: nn.Module, headsDict={'basic': None}):
        """

        Args:
            model: the native pytorch model
        Returns
            Turn model into a mtl_model, and change internal structure as necessary
        """
        super(mtl_model, self).__init__()

        self.headsDict = headsDict
        self.taskList = list(self.headsDict.keys())

        # Step 1: change every Conv2d Node into Conv2dNode to enable NAS
        self.model = self.embed_nas(model)

        # Step 2: compute the depth for each node, as it's needed for alter_train
        self.compute_depth()

    def compute_depth(self):
        """
            compute the depth for each Conv2dNode
        """
        pass

    def forward(self, x, stage='mtl', task=None, tau=5, hard=False, policy_idx=None):
        feature = self.model(x, stage, task, tau, hard, policy_idx)

        if task != None:
            output = self.headsDict[task](feature)
            return output
        else:
            warnings.warn('No task specified. Return feature.')
            return feature

    def embed_nas(self, model):
        for name, module in deepcopy(model).named_modules():
            print(module)
            if isinstance(module, nn.Conv2d):
                model.__setattr__(name, Conv2dNode(module.in_channels, module.out_channels,
                           module.kernel_size, module.stride,
                           module.padding, module.padding_mode,
                           module.dilation, bias=module.bias,
                           groups=module.groups, taskList=self.taskList))
            if isinstance(module, nn.BatchNorm2d):
                model.__setattr__(name, BN2dNode(module.num_features, module.eps, module.momentum,
                                  module.affine, module.track_running_stats,
                                  self.taskList))
            self.embed_nas(model)
        return model

    def share_bottom_policy(self, share_num):
        count = 0
        for node in self.modules():
            if isinstance(node, Conv2dNode):
                if count == share_num:
                    break
                else:
                    count += 1
                    for task in self.taskList:
                        node.policy[task] = nn.Parameter(torch.tensor([1., 0., 0.]))
                        node.policy[task].requires_grad = False
        return

    def max_node_depth(self):
        max_depth = 0
        for x in self.children():
            if isinstance(x, Conv2dNode):
                max_depth = max(x.depth, max_depth)
        return max_depth

    def policy_reg(self, task, policy_idx=None, tau=5, scale=1):
        """

        Args:
            task: current stage
            policy_idx: index of policy
            tau:
            scale:

        Returns:
            regulate the policy
        """
        reg = torch.tensor(0)
        if policy_idx is None:
            # Regularization for all policy
            for node in self.modules():
                if isinstance(node, Conv2dNode):
                    print(node)
                    policy_task = node.policy[task]
                    # gumbel_softmax make sure that we randomly pick each path while training
                    # e.g. if there is a 0.1 chance that our policy choose basic operator,
                    # and 0.9 chance to choose task specific path. Then ordinary probablity model ensure
                    # everytime we will go with task specifc path, and thus didn't train basic operator param
                    # however gumbel_softmax helps prevent that by choosig basic operator in certain rate.
                    possiblity = nn.functional.gumbel_softmax(policy_task, tau=tau, hard=False)
                    ##########################################################################
                    # Reg design1: l1 = sigmoid(g(b) - g(a)) * g(b) + sigmoid(g(c) - g(a)) * g(c)
                    #         l2 = sigmoid(g(b) - g(a)) * g(b) + sigmoid(g(b) - g(c)) * g(b)
                    #                     l1 = torch.sigmoid((possiblity[1]-possiblity[0])*scale).detach() * possiblity[1] + torch.sigmoid((possiblity[2]-possiblity[0])*scale).detach() * possiblity[2]
                    #                     l2 = (torch.sigmoid((possiblity[1]-possiblity[0])*scale).detach() + torch.sigmoid((possiblity[1]-possiblity[2])*scale).detach()) * possiblity[1]
                    #                     weight = 0.01 + (0.99-0.01) / self.max_node_depth() * node.depth
                    #                     reg = reg + (1-weight) * l1 + weight * l2
                    ##########################################################################
                    # Reg design2: loss = e^ (g(b) - g(a)) + e^(g(c) - g(a))
                    #                     loss = torch.exp(scale * (possiblity[1]-possiblity[0])) + torch.exp(scale * (possiblity[2]-possiblity[0]))
                    ##########################################################################
                    # Reg design3: ln(1+e ^ (g(b) - g(a))) + ln(1+e ^ (g(c) - g(a)))
                    loss = torch.log(1 + torch.exp(scale * (possiblity[1] - possiblity[0]))) + torch.log(
                        1 + torch.exp(scale * (possiblity[2] - possiblity[0])))
                    weight = (self.max_node_depth() - node.depth) / self.max_node_depth()
                    reg = reg + weight * loss
        elif policy_idx < self.max_policy_idx():
            # Regularization for current trained policy
            reg = self.current_policy_reg(policy_idx, task, tau, scale)
        return reg
