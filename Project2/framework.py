from typing import ForwardRef
import torch

torch.set_grad_enabled(False)


class Module(object):
    def __init__(self):
        super().__init__()

    def forward(self, * input):
        raise NotImplementedError

    def backward(self, * gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Sequential(Module):  # TODO: do this bad boy
    def __init__(self, * modules):
        super().__init__()
        self.parameters = []
        self.modules = modules

    def forward(self, *input):
        return

    def backward(self):
        return

    def param(self):
        return self.parameters


class Linear(Module):
    def __init__(self, startDimension, endDimension, reLu=False):
        super().__init__()
        self.startDimension = startDimension
        self.endDimension = endDimension
        self.tensor = torch.empty(1, 1)
        # TODO : change those values
        # If we use tanh as activation function, it is better to initialize bias and weights using xavier
        if reLu:
            self.w = 0
            self.b = 0
            self.db = 0
            self.dw = 0
        # If we use ReLu as the activation function, we initialize with He
        else:
            self.w = 0
            self.b = 0
            self.db = 0
            self.dw = 0

    def forward(self, *input):
        # Chapter 3.6 slide 9 : forward is x*w + b
        self.tensor = input
        return input.mm(self.w) + self.b

    def backward(self, *gradwrtoutput):
        """
        Scheme chapter 3.6 slide 9 :
        dl/dw = dl/ds * x^T
        dl/db = dl/ds
        """
        self.dw = self.tensor.t().mm(gradwrtoutput)
        self.db = gradwrtoutput.mean(0)  # TODO D'où sort ce putain de mean ?!?

        # Scheme chapter 3.6 slide 8 : backward is w^T * (dl/dx)
        # return gradwrtoutput.mm(self.w.T())
        return gradwrtoutput.matmul(self.w.T())

    def param(self):
        return


class ReLu(Module):
    def __init__(self):
        super().__init__()
        self.tensor = torch.empty(1, 1)

    def forward(self, input):
        # max(0,input)
        self.tensor = input
        return torch.max(torch.zeros(input.size()), input)

    def backward(self, *gradwrtoutput):
        # Derivative of max(0,input) is 0 if input <= 0, else it is 1
        if self.tensor <= 0:
            return torch.tensor([[0.]])
        return gradwrtoutput


class TanH(Module):
    def __init__(self):
        super().__init__()
        self.tensor = torch.empty(1, 1)

    def forward(self, input):
        # tanh(input)
        self.tensor = input
        return input.tanh()

    def backward(self, *gradwrtoutput):
        # Derivative of tanh(input) is 1/cosh^2(input)
        return gradwrtoutput * (1/torch.square(torch.cosh(self.tensor)))


class LossMSE(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, prediction):
        return torch.sum(torch.square(input - prediction)) # hot labels ??

    def backward(self, input, prediction):
        return 2 * (input - prediction)
