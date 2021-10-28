import torch
import torch.nn as nn

from .base_node import InputNode
from .layer_node import *

from proto import pytorch_pb2
import google.protobuf.text_format
import sys
import warnings

class MTLModel(nn.Module):
    def __init__(self, prototxt, headsDict={'basic': None}, BNsp=True):
        super(MTLModel, self).__init__()
        self.prototxt = self.parse_prototxt(prototxt)
        self.taskList = list(headsDict.keys())
        self.headsDict = headsDict
        self.BNsp = BNsp
        
        # Step 1: Generate computational graph from the backbone model prototxt
        #         self.net: the list of CNodes is used to represent the backbone model
        #         self.inputNode: store input data; used as the father node of the 1st layer
        self.net = nn.ModuleList()
        self.inputNode = None
        self.generate_computational_graph()
        
    def parse_prototxt(self, prototxt):
        # Function: Parse prototxt by protobuf
        net = pytorch_pb2.NetParameter()
        f = open(prototxt, 'r')
        net = google.protobuf.text_format.Merge(str(f.read()), net)
        f.close()
        return net
    
    def generate_computational_graph(self):
        # Function: Generate computational graph
        #           Iterate through all layers in the prototxt
        for i in range(0, len(self.prototxt.layer)):
            protoLayer = self.prototxt.layer[i]

            # Step 1: Find fatherNodes: CNode whose top name is the bottom name needed
            #         Handle with the 1st layer differently with InputNode
            fatherNodeList = []
            if protoLayer.bottom == ['data'] or protoLayer.bottom == ['blob1']:
                self.inputNode = InputNode(self.prototxt.input_dim[1])
                fatherNodeList.append(self.inputNode)
            else:
                for topName in protoLayer.bottom:
                    fatherNodeList.append(self.find_CNode_by_top_name(topName))
            
            # Step 2: Generate CNode
            CNode = self.generate_CNode(protoLayer, fatherNodeList)
            self.net.append(CNode)
        return
                    
    def generate_CNode(self, protoLayer, fatherNodeList):
        # Function: Generate CNode for different types of protolayer
        #           e.g., Conv2dNode, BN2dNode, ReLUNode, PoolNode, EltNode, DropoutNode, LinearNode
        nodeType = protoLayer.type

        if nodeType == 'Convolution':
            CNode = Conv2dNode(protoLayer, fatherNodeList, self.taskList)
        elif nodeType == 'BatchNorm':
            CNode = BN2dNode(protoLayer, fatherNodeList, self.taskList, self.BNsp)
        elif nodeType == 'ReLU':
            CNode = ReLUNode(protoLayer, fatherNodeList, self.taskList)
        elif nodeType == 'Pooling':
            CNode = PoolNode(protoLayer, fatherNodeList, self.taskList)
        elif nodeType == 'Eltwise':
            CNode = EltNode(protoLayer, fatherNodeList, self.taskList)
        elif nodeType == 'Dropout':
            CNode = DropoutNode(protoLayer, fatherNodeList, self.taskList)
        elif nodeType == 'InnerProduct':
            CNode = LinearNode(protoLayer, fatherNodeList, self.taskList)
        else:
            # Quit and Warning
            sys.exit(nodeType + ': Wrong Layer Type.')
        return CNode
        
    def find_CNode_by_top_name(self, topName):
        # Function: Return the CNode with the required top name
        #           Note to find the latest one
        CNode = None
        for node in self.net:
            if node.layerTop == topName:
                CNode = node
        if CNode == None:
            sys.exit('No such CNode with required top name: ' + topName)
        return CNode
        
    def forward(self, x, stage='mtl', task=None, tau=5, hard=False, policy_idx=None):
        # Step 1: Obtain features from the backbone model
        self.inputNode.set_data(x)
        for node in self.net:
            node.output = node(stage, task, tau, hard)
        feature = node.output
        
        # Step 2: Add heads for each task
        if task != None:
            output = self.headsDict[task](feature)
            return output
        else:
            warnings.warn('No task specified. Return feature.')
            return feature
    
    # Helper functions
    # For policy regularization
    def max_node_depth(self):
        return max([x.depth for x in self.net])
    
    def current_policy_reg(self, current, task, tau, scale=6):
        policy_task = self.current_policy(current, task)
        possiblity = nn.functional.gumbel_softmax(policy_task, tau=tau, hard=False)
        weight = torch.sigmoid((possiblity[1]-possiblity[0])*scale).detach() + torch.sigmoid((possiblity[1]-possiblity[2])*scale).detach()
        reg = weight * possiblity[1]
        return reg
    
    def policy_reg(self, task, policy_idx=None, tau=5, scale=1):
        reg = torch.tensor(0)
        if policy_idx is None: 
            # Regularization for all policy
            for node in self.net:
                if node.taskSp and not node.assumeSp:
                    policy_task = node.policy[task]
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
                    loss = torch.log(1+torch.exp(scale * (possiblity[1]-possiblity[0]))) + torch.log(1+torch.exp(scale * (possiblity[2]-possiblity[0])))
                    weight = (self.max_node_depth() - node.depth) / self.max_node_depth()
                    reg = reg + weight * loss
        elif policy_idx < self.max_policy_idx():
            # Regularization for current trained policy 
            reg = self.current_policy_reg(policy_idx, task, tau, scale)
        return reg
    
    # For bottom-shared policy
    def share_bottom_policy(self, share_num):
        count = 0
        for node in self.net:
            if count == share_num:
                break
            if node.taskSp and not node.assumeSp:
                count += 1
                for task in self.taskList:
                    node.policy[task] = nn.Parameter(torch.tensor([1., 0., 0.]))
                    node.policy[task].requires_grad = False
        return