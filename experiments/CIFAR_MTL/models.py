from typing import Iterator

import torch
import torch.nn as nn


class SimpleConvNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, activation):
        nn.Module.__init__(self)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            padding=1,
            padding_mode="reflect",
        )
        self.act = nn.ReLU(inplace=True) if activation == 'relu' else \
                   nn.LeakyReLU(inplace=True) if activation == 'leaky_relu' else \
                   nn.ELU(inplace=True)
        self.maxpool = nn.MaxPool2d(3)

    def forward(self, x):
        y = self.conv(x)
        y = self.act(y)
        y = self.maxpool(y)
        return y


class CNN(nn.Module):
    def __init__(self, n_tasks, linear_task_head=True, activation='elu'):
        super().__init__()
        self.n_tasks = n_tasks
        self.linear_task_head = linear_task_head
        self.last_shared = SimpleConvNetBlock(160, 160, 3, activation)
        self.shared = nn.Sequential(
            SimpleConvNetBlock(3, 160, 3, activation),
            SimpleConvNetBlock(160, 160, 3, activation),
            self.last_shared,
            nn.BatchNorm2d(160),
        )

        self._init_task_heads()

    def _init_task_heads(self):
        for i in range(self.n_tasks):
            task_specific = nn.Linear(160, 1) if self. linear_task_head \
                else nn.Sequential(
                        nn.Linear(160, 320),
                        nn.ReLU(),
                        nn.Linear(320, 320),
                        nn.ReLU(),
                        nn.Linear(320, 1),
                    )
            setattr(self, f"head_{i}", task_specific)
        self.task_specific = nn.ModuleList(
            [getattr(self, f"head_{i}") for i in range(self.n_tasks)]
        )

    def forward(self, x, return_representation=False):
        features = self.shared(x)
        features = features.flatten(start_dim=1)
        logits = self.forward_task_heads(features)
        if return_representation:
            return logits, features
        return logits

    def forward_task_head(self, features, t):
        return getattr(self, f"head_{t}")(features)

    def forward_task_heads(self, features):
        return torch.cat(
            [getattr(self, f"head_{i}")(features) for i in range(self.n_tasks)], dim=1
        )

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.shared.parameters()

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.task_specific.parameters()

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return (p for n, p in self.named_parameters() if "last_shared" in n)


class LinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(160, 1)

    def forward(self, x):
        return self.layer(x)

    def init_params(self, params):
        self.layer.weight.data = params[0]
        self.layer.bias.data = params[1]


if __name__ == '__main__':
    net = CNN(3)
    logits, features = net(torch.randn((2, 3, 32, 32)), return_representation=True)
    grads = torch.randn((2, 160))
    features.backward(gradient=grads)
    aaa = list(net.task_specific_parameters())
    bbb = 1

    net2 = LinearNet()
    net2.init_params(tuple(aaa[:2]))
