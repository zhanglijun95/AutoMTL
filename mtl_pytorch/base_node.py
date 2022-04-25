import torch.nn as nn
import torch
import copy
import sys
# Author: yiming huang

# from framework.layer_containers import LazyLayer

class BasicNode(nn.Module):

    def __init__(self, taskList=['basic']):
        """ define the basic node/layer in AutoMTL

        Args:
            taskList: A list of Task that the model wants to achieve
        """
        super().__init__()
        self.taskList = taskList  # task list: task-specific operator()
        self.taskOp = nn.ModuleDict()  # task operator, i.e a copy of basic operator
        self.basicOp = None
        self.depth = 0
        self.policy = nn.ParameterDict()

    def build_layer(self):
        """

        Returns:
            pipeline for generating AutoMTL computation Node
        """
        # Step 1: Generate Basic operators depend on the user's input
        self.generate_basicOp()

        # Step 2: Generate Task specific Operator based on basic operator
        self.generate_taskOp()

        return

    def generate_basicOp(self):
        """

        Returns:
            initialize basic operator based on specific layer
        """
        return

    def generate_taskOp(self):
        """

        Returns:
            initialize task specific operator by coping basic operator
        """
        return

    def forward(self, x, stage='common', task=None, tau=5, hard=False):
        """
        Args:
            x: input data
            stage: the stage for training
            task: the task list
            tau:
            hard: whether to share the common parameter

        Returns:
            the result of forward propagation
        """
        if stage == 'common':
            return self.compute_common(x)
        elif stage == 'hard_sharing':
            return self.compute_hard_sharing(x)
        elif stage == 'task_specific':
            if task is not None:
                return self.compute_task_weights(x, task)
            else:
                sys.exit('Please enter the specified task for stage==' + stage)
        elif stage == 'combined' or stage == 'pre_train_all':
            if task is not None:
                return self.compute_combined(x, task)
            else:
                sys.exit('Please enter the specified task for stage==' + stage)
        elif stage == 'mtl':
            if len(self.taskList) > 1:
                return self.compute_mtl(x, task, tau, hard)
            else:
                sys.exit('Only 1 task in the multi-task model. Please try stage="common".')
        else:
            sys.exit('No forward function for the given stage.')

    def compute_common(self, x):
        """

        Args:
            x: input data
        Returns: compute the output of basic operator

        """
        return self.basicOp(x)

    def compute_hard_sharing(self, x):
        """

        Args:
            x: input data
        Returns: compute the output for hard parameter sharing training

        """
        return self.compute_common(x)

    def copy_weight_after_pretrain(self):
        """

        Returns: copy the shared operator to
        other task-specific operator for initialization

        """
        if len(self.taskList) > 1 and self.taskSp:
            for task in self.taskList:
                self.taskOp[task] = copy.deepcopy(self.baiscOp)
        return

    def compute_specific(self, x, task):
        return self.taskOp[task](x)

    def compute_task_weights(self, x, task):
        """

        Args:
            x: input data
            task: the task we want to get for result
        Returns: the weights for task we want to compute

        """
        return self.compute_specific(x, task)

    def compute_mtl(self, x, task, tau=5, hard=False):
        """

        Args:
            x: input data
            task: the current stage
            tau:
            hard: whether to use hard sharing

        Returns:
            output
        """
        if self.taskSp:
            # Conv2d or BN
            if self.assumeSp:
                # BN only consider task specific
                return self.compute_specific(x, task)
            else:
                # Conv2d
                policy_task = self.policy[task]
                if hard is False:
                    # Policy-train
                    # possibility of each task
                    possiblity = nn.functional.gumbel_softmax(policy_task, tau=tau, hard=hard)
                    feature_common = self.compute_common(x)
                    feature_specific = self.compute_specific(x, task)
                    feature_downsample = self.compute_downsample(x, task)
                    feature = feature_common * possiblity[0] + feature_specific * possiblity[1] + feature_downsample * \
                              possiblity[2]
                else:
                    # Post-train or Validation
                    branch = torch.argmax(policy_task).item()
                    if branch == 0:
                        feature = self.compute_common(x)
                    elif branch == 1:
                        feature = self.compute_specific(x, task)
                    elif branch == 2:
                        feature = self.compute_downsample(x, task)
                return feature
        else:
            return self.compute_common(x)


