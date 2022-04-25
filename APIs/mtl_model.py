from operator import mod
import warnings
from layer_node import Conv2dNode, BN2dNode, BasicNode

import torch.nn as nn
import torch
from mtl_pytorch.base_node import BasicNode
from copy import deepcopy

class mtl_model(nn.Module):

    def __init__(self):
        """
            compute depth for apply policy regularization
        """
        super(mtl_model, self).__init__()

    def compute_depth(self):
        """
            compute the depth for each Basic node
        """
        cur_dep = 0
        for module in self.modules():
            if self.check_type(module):
                module.depth = cur_dep
                cur_dep += 1

    def share_bottom_policy(self, share_num):
        count = 0
        for node in self.modules():
            if self.check_type(node):
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
        for module in self.modules():
            if self.check_type(module):
                max_depth = max(module.depth, max_depth)
        return max_depth

    def check_type(self, module):
        return isinstance(module, Conv2dNode) or isinstance(module, BN2dNode)

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
            for module in self.modules():
                if isinstance(module, Conv2dNode):
                    policy_task = module.policy[task]
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
                    weight = (self.max_node_depth() - module.depth) / self.max_node_depth()
                    reg = reg + weight * loss
        elif policy_idx < self.max_policy_idx():
            # Regularization for current trained policy
            reg = self.current_policy_reg(policy_idx, task, tau, scale)
        return reg
