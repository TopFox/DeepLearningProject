import torch

torch.set_grad_enabled(False)


class Module (object):
    def __init__(self):
        super().__init__()

    def forward(self, * input):
        raise NotImplementedError

    def backward(self, * gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Sequential(Module):
    def __init__(self):
        super().__init__()

    def forward(self, * input):
        return

    def backward(self):
        return

    def param(self):
        return
