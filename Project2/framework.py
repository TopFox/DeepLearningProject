# Project 2
# Coded by : Arnaud Savary and Jérémie Guy

# --------------------------------------- Imports --------------------------------------

from typing import ForwardRef
import torch
import math

# As per the project instructions, we deactivate the autograd option
torch.set_grad_enabled(False)

# --------------------------------------- Module Structure --------------------------------------
# We will define here the basic structure of every module that will constitute our model

# Defining class Module that will be implemented by every subclass


class Module(object):
    def __init__(self):
        super().__init__()

    # Mendatory methods : every module should implement forward and backward - thus the error
    # method to pass the parameters to the next layer during the forward pass
    def forward(self, * input):
        raise NotImplementedError

    # method to pass the parameters to the previous layer during the backpropagation
    def backward(self, * gradwrtoutput):
        raise NotImplementedError

    # optional : only some of the modules need to implement the method - thus the return None
    # method to return the parameters
    def param(self):
        return None

    # method to put tha gradients back to zero before the backpropagation
    def zero_grad(self):
        return None

# --------------------------------------- Model Structure --------------------------------------
# We now need to define how our network is entwined and links every module and their respective methods together

# Sequential Module : this allows one to create the sequence of modules and execute the structure of the model that we want


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        # defining the basic parameters of our model
        self.name = 'Sequential'
        self.parameters = []
        self.modules = modules
        self.sequence = []

        # in the case of the Linear module in the model, we need to fetch the parameters (used in forward pass and backpropagation)
        for module in self.modules:
            if module.name == 'Linear':
                for parameter in module.param():
                    self.parameters.append(parameter)

    #  Method to transmit back the parameters from the current to the next layer
    def forward(self, input):
        for index, module in enumerate(self.modules):
            if module.name == 'Linear':
                module.w = self.param()[index][0]
                module.b = self.param()[index+1][0]
            input = module.forward(input)
            self.sequence.append(module.tensor)
        return input

    # Method to transmit back the parameters from the next layer to the current layer
    def backward(self, gradwrtoutput):

        # getting the reverse order of modules as we change the propgation direction
        reversedModules = self.modules[::-1]
        reversedSequence = self.sequence[::-1]
        reversedParamaters = []

        # for every module we call its backward method so it computes the gradient (with respect to the output)
        # and the new parameters that have to be transmitted to the previous layer
        for index, module in enumerate(reversedModules):
            # getting the correct module
            module.tensor = reversedSequence[index]
            gradwrtoutput = module.backward(
                gradwrtoutput)  # computing the gradiant
            if module.name == 'Linear':  # for the Linear module, get the updated parameters
                for parameter in module.param()[::-1]:
                    reversedParamaters.append(parameter)

        # updating the new parameters with the ones fetched from backwards
        self.parameters = reversedParamaters[::-1]

    # Method to fetch the parameters (used for Linear)
    def param(self):
        return self.parameters

    # Method to put back the gradient to zero : this will be used before backpropagating as we need to compute the new parameters
    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

# --------------------------------------- Hidden Layer  --------------------------------------

# Linear Module : this modules allows us to change the number of nodes (input/ouput) inside the layer. Each of the nodes are
# weighted and biased - the computation of these weights and bias will then allow the model to make predictions with a given input


class Linear(Module):
    def __init__(self, startDimension, endDimension, activation_function='TanH'):
        super().__init__()
        self.name = 'Linear'

        self.startDimension = startDimension
        self.endDimension = endDimension
        # this will be the tensor used for the computations of the weights and biases during training
        self.tensor = torch.empty(1, 1)

        ##### initalizing the wights and biases #####

        # There are a few methods of initializaion for the wieghts and biases. As we use ReLU and TanH for our
        # modules, we used two different methods of initializaion that have an impact on the standard deviation
        # depending on the module :

        # If we use tanh as activation function, it is better to initialize bias and weights using Xavier
        if activation_function == 'TanH':
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
        # Using the formula in Scheme chapter 3.6 slide 9 : forward is x*w + b
        self.tensor = input
        return input.mm(self.w) + self.b

    def backward(self, gradwrtoutput):
        # Using the formula in Scheme chapter 3.6 slide 9, we have :
        # dl/dw = dl/ds * x^T
        # dl/db = dl/ds
        self.dw = self.tensor.t().mm(gradwrtoutput)
        self.db = gradwrtoutput

        # Using the formula in Scheme chapter 3.6 slide 8 we have : backward is w^T * (dl/dx)
        return gradwrtoutput.mm(self.w.t())

    # sending the parameters (here list of dimension 2 to easily get w,dw or b,db)
    def param(self):
        return [[self.w, self.dw], [self.b, self.db]]

    # Putting the gradient back to zero to be able to recompute it using new parameters
    def zero_grad(self):
        self.dw.zero_()
        self.db.zero_()

# --------------------------------------- Activation Functions --------------------------------------
# Here we define the activation functions which will allow us to transition between two hidden layers of our model.


# Method ReLU : its a method which sends the output of a layer as the input of the next layer as max(0,input)
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

# Method TanH : very similar to ReLU put using the TanH (and its derivative) function(s) to send inputs to the next layer


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

# --------------------------------------- Error criterion --------------------------------------
# Function to compute the error of the model. This is the function we use during the test process to
# determine how good the model fares and if it learns correctly.

# In the Mean Squared Error we compute the error between the target values. In our case we need to classify if a point
# is inside and outside our circle respecively as 1 and 0.


class LossMSE(Module):
    def __init__(self):
        super().__init__()
        self.name = 'LossMSE'

    # Forward pass crieterion error : sum((input - prediction)^2)/N - we do the sum as we want the mean global error
    # for every point and not just for one specific point : this error reflects the whole model performance
    def forward(self, input, prediction):
        prediction = torch.reshape(prediction, (len(input), 1))
        return 1/len(input)*torch.sum(torch.square(input.sub(prediction)))

    # Derivative of the forward pass criterion error : we don't need any more sum as the first MSE backward gives
    def backward(self, input, prediction):
        prediction = torch.reshape(prediction, (len(input), 1))
        return 2 * input.sub(prediction)

# --------------------------------------- Optimizer --------------------------------------
# The optimizer allows the nodel to do itslef some fine-tuning on the parameters during the training process

# Stochastic Gradient descent allows to fine tune the weights and bias parameters during the backpropagation


class SGD():
    def __init__(self, parameters, lr, wd):
        self.name = 'SGD'
        self.parameters = parameters
        self.lr = lr
        self.wd = wd

    # Tuning the parameters using, if activated, the weight decay (wd) : it reduces the impact of
    # of a parameters and slows over time the learning proces. This is to counter overfitting
    def step(self, parameters):
        for parameter in parameters:
            parameter[0] = parameter[0].sub(self.lr * parameter[1])
            if self.wd:
                parameter[0] -= 2*self.wd*parameter[0]
