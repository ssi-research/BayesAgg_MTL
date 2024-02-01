"""from Senser and Koltun git repo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterator
from itertools import chain

"""
Adapted from: https://github.com/nik-dim/pamal/blob/master/src/models/factory/resnet.py
"""

class BasicBlock(nn.Module):
    """BasicBlock block for the Resnet. Adapted from official Pytorch source code."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetEncoder(nn.Module):
    """
    ResNet architecture adapted from official Pytorch source code.
    The main difference lies in replacing the last FC layer dedicated for classification
    with a final FC layer that will be the shared representation for MTL
    """

    def __init__(self, n_tasks=3, num_blocks=(2, 2, 2, 2),
                 task_outputs=(1, 2, 5), in_channels=3, activation="elu"):
        super(ResNetEncoder, self).__init__()

        self.n_tasks = n_tasks
        self.in_planes = 64
        self.in_channels = in_channels
        self.task_outputs = task_outputs
        self.activation = activation

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, 256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._init_task_heads()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_task_heads(self):
        for i in range(len(self.task_outputs)):
            setattr(self, f"head_{i}", torch.nn.Linear(256, self.task_outputs[i]))
        self.task_specific = torch.nn.ModuleList(
            [getattr(self, f"head_{i}") for i in range(self.n_tasks)]
        )

    def forward(self, x, return_representation=False):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        features = F.elu(self.linear(out)) if self.activation.lower() == "elu" else F.relu(self.linear(out))
        logits = [getattr(self, f"head_{i}")(features) for i in range(self.n_tasks)]
        if return_representation:
            return logits, features
        return logits

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return chain(
            self.conv1.parameters(),
            self.bn1.parameters(),
            self.layer1.parameters(),
            self.layer2.parameters(),
            self.layer3.parameters(),
            self.layer4.parameters(),
            self.linear.parameters()
        )

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.task_specific.parameters()

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.linear.parameters()


if __name__ == '__main__':
    net = ResNetEncoder()
    net(torch.randn((5, 3, 128, 128)))