# Project 2
# Coded by : Arnaud Savary and Jérémie Guy

# --------------------------------------- Imports --------------------------------------

import framework
from framework import *
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------- Generating Data --------------------------------------
# We first need to generate the data on which the model work


def generateData(pointsNumber, plotPoints=True):
    # Generating the training and testing points randomely and computing the class the belong to i.e. checking if their distance to the
    # center of the circle is smaller than 1/sqrt(2*pi)
    train_input = torch.rand(pointsNumber, 2)
    train_target = (train_input.subtract(0.5).square().sum(
        1).sqrt() <= (1/math.sqrt((2*math.pi)))).long()
    test_input = torch.rand(pointsNumber, 2)
    test_target = (test_input.subtract(0.5).square().sum(
        1).sqrt() <= (1/math.sqrt((2*math.pi)))).long()

    # Plotting the training data points with reference to the circle they should belong in
    if plotPoints:
        plt.title('Train data')
        plt.gca().add_patch(plt.Circle((0.5, 0.5), 1 /
                                       math.sqrt((2*math.pi)), color='k', alpha=0.2))
        for i in range(pointsNumber):
            if train_target[i] == 0:
                plt.plot(train_input[i][0], train_input[i][1], 'ro')
            else:
                plt.plot(train_input[i][0], train_input[i][1], 'go')
        plt.show()
        plt.close()

        # Plotting the training data points with reference to the circle they should belong in
        plt.title('Test data')
        plt.gca().add_patch(plt.Circle((0.5, 0.5), 1 /
                                       math.sqrt((2*math.pi)), color='k', alpha=0.2))
        for i in range(pointsNumber):
            if test_target[i] == 0:
                plt.plot(test_input[i][0], test_input[i][1], 'ro')
            else:
                plt.plot(test_input[i][0], test_input[i][1], 'go')
        plt.show()

    # the function returns the needed data
    return train_input, train_target, test_input, test_target


def train_model(model, criterion, train_input, train_target, test_input, test_target, mini_batch_size=50, nb_epochs=250, lr=5e-3, wd=1e-6, mustPrint=True, plotLoss=True, plotPoints=True):
    optimizer = framework.SGD(model.param(), lr=lr, wd=wd)
    losses = []
    for e in range(nb_epochs):
        acc_loss = 0
        numberOfTrainErrors = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            train_output = model.forward(
                train_input.narrow(0, b, mini_batch_size))
            loss = criterion.forward(
                train_output, train_target.narrow(0, b, mini_batch_size))
            acc_loss += loss.item()

            model.zero_grad()
            model.backward(criterion.backward(
                train_output, train_target.narrow(0, b, mini_batch_size)))
            optimizer.step(model.param())

            if e == nb_epochs-1:
                for k in range(mini_batch_size):
                    train_output[k] = round(float((train_output[k] + 1)/2))
                    if train_target[b + k] != train_output[k]:
                        numberOfTrainErrors += 1

        acc_loss /= train_input.size(0)
        losses.append(acc_loss)

        test_prediction = []
        numberOfTestErrors = 0
        for b in range(0, test_input.size(0), mini_batch_size):
            test_output = model.forward(
                test_input.narrow(0, b, mini_batch_size))

            for k in range(mini_batch_size):
                test_output[k] = round(float(test_output[k]))
                if test_target[b + k] != test_output[k]:
                    numberOfTestErrors += 1
            if e == nb_epochs-1:
                test_prediction.extend(test_output)
        if e % 5 == 0 and mustPrint:
            print('Epoch:', e, ', number of errors in final test',
                  numberOfTestErrors)
    if plotLoss:
        plt.title('Evolution of loss')
        plt.plot(range(250), losses)
        plt.show()

    if plotPoints:
        for i in range(test_input.size(0)):
            if test_prediction[i] == 0:
                plt.plot(test_input[i][0], test_input[i][1], 'ro')
            else:
                plt.plot(test_input[i][0], test_input[i][1], 'go')
        plt.title('Predictions of the model')
        plt.gca().add_patch(plt.Circle((0.5, 0.5), 1 /
                                       math.sqrt((2*math.pi)), color='k', alpha=0.2))
        plt.show()

    return numberOfTestErrors


def model_tuning(number_of_runs=5):
    nb_epochs = 100
    mini_batch_size = 50
    lrs = [1e-3, 5e-3, 1e-2]
    wds = [False, 1e-6]
    activation_functions = [framework.ReLu(), framework.TanH()]
    best_validation_errors = 1000

    for lr in lrs:
        for wd in wds:
            for act_fun_1 in activation_functions:
                for act_fun_2 in activation_functions:
                    for act_fun_3 in activation_functions:
                        for act_fun_4 in activation_functions:
                            validation_errors = []
                            i = 0
                            while i < number_of_runs:
                                linear1 = framework.Linear(2, 25, tanh=False)
                                linear2 = framework.Linear(25, 25)
                                linear3 = framework.Linear(25, 25)
                                linear4 = framework.Linear(25, 1)
                                criterion = framework.LossMSE()
                                train_input, train_target, validation_input, validation_target = generateData(
                                    1000, plotPoints=False)
                                model = framework.Sequential(
                                    linear1, act_fun_1, linear2, act_fun_2, linear3, act_fun_3, linear4, act_fun_4)
                                validation_error = train_model(model, criterion, train_input, train_target, validation_input, validation_target,
                                                               mini_batch_size=mini_batch_size, nb_epochs=nb_epochs, mustPrint=False, plotLoss=False, plotPoints=False)
                                # TODO : Remove if
                                if validation_error < 450:
                                    i += 1
                                    validation_errors.append(validation_error)
                                else:
                                    print(validation_error)

                            print('Lr =', str(lr) + ', wd =', str(wd) + ', activation functions =',
                                  [act_fun_1.name, act_fun_2.name, act_fun_3.name, act_fun_4.name], ', mean validation error =', np.mean(validation_errors))

                            if np.mean(validation_errors) < best_validation_errors:
                                best_lr = lr
                                best_wd = wd
                                best_act_fun_1 = act_fun_1
                                best_act_fun_2 = act_fun_2
                                best_act_fun_3 = act_fun_3
                                best_act_fun_4 = act_fun_4
    print('--- Best parameters found ---')
    print('Lr =', str(best_lr) + ', wd =', str(wd) + ', activation functions =',
          [best_act_fun_1.name, best_act_fun_2.name, best_act_fun_3.name, best_act_fun_4.name], ', mean validation error =', best_validation_errors)
    return best_lr, best_wd, best_act_fun_1, best_act_fun_2, best_act_fun_3, best_act_fun_4


def test_framework():
    # Parameters
    nb_epochs = 250
    mini_batch_size = 50

    # Framework
    linear1 = framework.Linear(2, 25, tanh=False)
    linear2 = framework.Linear(25, 25)
    linear3 = framework.Linear(25, 25)
    linear4 = framework.Linear(25, 1)
    criterion = framework.LossMSE()

    # Data
    train_input, train_target, test_input, test_target = generateData(
        1000, plotPoints=False)

    # Model tuning
    best_lr, best_wd, best_act_fun_1, best_act_fun_2, best_act_fun_3, best_act_fun_4 = model_tuning()

    # Model
    model = framework.Sequential(
        linear1, best_act_fun_1, linear2, best_act_fun_2, linear3, best_act_fun_3, linear4, best_act_fun_4)

    # Train model
    _ = train_model(model, criterion, train_input, train_target, test_input,
                    test_target, mini_batch_size=mini_batch_size, nb_epochs=nb_epochs, lr=best_lr, wd=best_wd)


test_framework()
