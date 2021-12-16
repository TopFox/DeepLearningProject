from typing import ForwardRef
import torch
import math

torch.set_grad_enabled(False)


class Module(object):
    def __init__(self):
        super().__init__()

    def forward(self, * input):
        raise NotImplementedError

    def backward(self, * gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return None

    def zero_grad(self):
        return None


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.name = 'Sequential'
        self.parameters = []
        self.modules = modules
        self.sequence = []

        for module in self.modules:
            if module.name == 'Linear':
                for parameter in module.param():
                    self.parameters.append(parameter)

    # TODO understand this
    def forward(self, input):
        for index, module in enumerate(self.modules):
            if module.name == 'Linear':
                module.w = self.param()[index][0]
                module.b = self.param()[index+1][0]
            input = module.forward(input)
            self.sequence.append(module.tensor)
        return input

    def backward(self, gradwrtoutput):
        reversedModules = self.modules[::-1]
        reversedSequence = self.sequence[::-1]
        reversedParamaters = []

        for index, module in enumerate(reversedModules):
            module.tensor = reversedSequence[index]
            gradwrtoutput = module.backward(gradwrtoutput)
            if module.name == 'Linear':
                for parameter in module.param()[::-1]:
                    reversedParamaters.append(parameter)

        self.parameters = reversedParamaters[::-1]

    def param(self):
        return self.parameters

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()


class Linear(Module):
    def __init__(self, startDimension, endDimension, tanh=True):
        super().__init__()
        self.name = 'Linear'

        self.startDimension = startDimension
        self.endDimension = endDimension
        self.tensor = torch.empty(1, 1)
        # If we use tanh as activation function, it is better to initialize bias and weights using xavier
        if tanh:
            std = math.sqrt(1./(startDimension+endDimension))
        # If we use ReLu as the activation function, we initialize with He
        else:
            std = math.sqrt(2./(startDimension+endDimension))
        self.w = torch.empty(startDimension, endDimension).normal_(0, std)
        self.b = torch.empty(endDimension).normal_(0, std)
        self.dw = torch.zeros(startDimension, endDimension)
        self.db = torch.zeros(endDimension)
        self.tensor = torch.empty(1, 1)

    def forward(self, input):
        # Chapter 3.6 slide 9 : forward is x*w + b
        self.tensor = input
        return input.mm(self.w) + self.b

    def backward(self, gradwrtoutput):
        """
        Scheme chapter 3.6 slide 9 :
        dl/dw = dl/ds * x^T
        dl/db = dl/ds
        """
        self.dw = self.tensor.t().mm(gradwrtoutput)
        self.db = gradwrtoutput

        # Scheme chapter 3.6 slide 8 : backward is w^T * (dl/dx)
        return gradwrtoutput.mm(self.w.t())

    def param(self):
        return [[self.w, self.dw], [self.b, self.db]]

    def zero_grad(self):
        self.dw.zero_()
        self.db.zero_()


class ReLu(Module):
    def __init__(self):
        super().__init__()
        self.tensor = torch.empty(1, 1)
        self.name = 'ReLu'

    def forward(self, input):
        # max(0,input)
        self.tensor = input
        return torch.max(input, torch.zeros(input.size()))

    def backward(self, gradwrtoutput):
        # Derivative of max(0,input) is 0 if input <= 0, else it is 1
        return gradwrtoutput*(self.tensor > 0).float()


class TanH(Module):
    def __init__(self):
        super().__init__()
        self.tensor = torch.empty(1, 1)
        self.name = 'TanH'

    def forward(self, input):
        # tanh(input)
        self.tensor = input
        return input.tanh()

    def backward(self, gradwrtoutput):
        # Derivative of tanh(input) is 1/cosh^2(input)
        return gradwrtoutput * (1/torch.square(torch.cosh(self.tensor)))


class LossMSE(Module):
    def __init__(self):
        super().__init__()
        self.name = 'LossMSE'

    def forward(self, input, prediction):
        prediction = torch.reshape(prediction, (len(input), 1))
        return torch.sum(torch.square(input.sub(prediction)))

    def backward(self, input, prediction):
        prediction = torch.reshape(prediction, (len(input), 1))
        return 2 * input.sub(prediction)


class SGD():
    def __init__(self, parameters, lr, wd):
        self.name = 'SGD'
        self.parameters = parameters
        self.lr = lr
        self.wd = wd

    def step(self, parameters):
        for parameter in parameters:
            parameter[0] = parameter[0].sub(self.lr * parameter[1])
            if self.wd:
                parameter[0] -= 2*self.wd*parameter[0]
