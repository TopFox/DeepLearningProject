# Project 2
# Coded by : Arnaud Savary and Jérémie Guy

# --------------------------------------- Imports --------------------------------------

import framework
from framework import *
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------- Generating Data --------------------------------------
# We first need to generate the data on which the model work

# We nee to generate 1000 points that have two coordinates between 0 and 1 using a normal distribution.
# The main point of the algorithm is to predict if any given point is inside of a circle of radius
# 1/sqrt(2*pi)


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
                # points inside the circle
                plt.plot(train_input[i][0], train_input[i][1], 'ro')
            else:
                # points outside the circle
                plt.plot(train_input[i][0], train_input[i][1], 'go')
        plt.show()

        # Plotting the training data points with reference to the circle they should belong in
        plt.title('Test data')
        plt.gca().add_patch(plt.Circle((0.5, 0.5), 1 /
                                       math.sqrt((2*math.pi)), color='k', alpha=0.2))
        for i in range(pointsNumber):
            if test_target[i] == 0:
                # points inside the circle
                plt.plot(test_input[i][0], test_input[i][1], 'ro')
            else:
                # points outside the circle
                plt.plot(test_input[i][0], test_input[i][1], 'go')
        plt.show()

    # the function returns the needed data
    return train_input, train_target, test_input, test_target

# --------------------------------------- Training Model --------------------------------------

# Main function to train the model to our problem


def train_model(model, criterion, train_input, train_target, test_input, test_target, mini_batch_size=50, nb_epochs=250, lr=5e-3, wd=1e-6, mustPrint=False, plotLoss=False, plotPoints=False):
    # initializing the optimizer
    optimizer = framework.SGD(model.param(), lr=lr, wd=wd)
    losses = []

    # Training for every epoch
    for e in range(nb_epochs):
        acc_loss = 0  # accumulated loss
        numberOfTrainErrors = 0

        # We iteratively train using the batch size
        for b in range(0, train_input.size(0), mini_batch_size):

            # runing the model and fetching its output
            train_output = model.forward(
                train_input.narrow(0, b, mini_batch_size))
            # computing the training loss
            loss = criterion.forward(
                train_output, train_target.narrow(0, b, mini_batch_size))
            acc_loss += loss.item()

            # Putting back the gradient to zero before computing the new one during backpropagation
            model.zero_grad()

            # computing said backpropagation
            model.backward(criterion.backward(
                train_output, train_target.narrow(0, b, mini_batch_size)))

            # optimizing the parameters with the learning rate (and, if activated, weight decay)
            optimizer.step(model.param())

            # At the final epoch, we test the trained model for every batch on the train set
            if e == nb_epochs-1:
                for k in range(mini_batch_size):
                    # rounding the results to compute the output class prediction
                    test_output[k] = round(float(test_output[k]))
                    # if we have a wrong prediction, increment the error count
                    if train_target[b + k] != train_output[k]:
                        numberOfTrainErrors += 1

        # Normalizing the accumulated loss and storing it
        acc_loss /= train_input.size(0)
        losses.append(acc_loss)

        # At each epoch we want to see the evolution of the model performances on the test set
        test_prediction = []
        numberOfTestErrors = 0

        # We test the model for every batch on the test data
        for b in range(0, test_input.size(0), mini_batch_size):
            # runing the model and fetching its output
            test_output = model.forward(
                test_input.narrow(0, b, mini_batch_size))

            for k in range(mini_batch_size):
                # rounding the results to compute the output class prediction
                test_output[k] = round(float(test_output[k]))
                # if we have a wrong prediction, increment the error count
                if test_target[b + k] != test_output[k]:
                    numberOfTestErrors += 1
            # At the last epoch we add the final test output to the model prediction
            if e == nb_epochs-1:
                test_prediction.extend(test_output)

        # Every 5 epoch print the test error
        if e % 5 == 0 and mustPrint:
            print('Epoch:', e, ', percentage of errors in final test {:.2f}'.format(
                numberOfTestErrors/len(test_input)*100), '%')

    if plotLoss:
        # Plotting the loss
        plt.title('Evolution of loss')
        plt.ylabel('accumulated loss')
        plt.xlabel('Epochs')

        plt.plot(range(nb_epochs), losses)
        plt.show()

    # Plotting the visual representation of the output values
    if plotPoints:
        for i in range(test_input.size(0)):
            if test_prediction[i] == 0:
                # points that are predicted as outside the circle
                plt.plot(test_input[i][0], test_input[i][1], 'ro')
            else:
                # points that are predicted as inside the circle
                plt.plot(test_input[i][0], test_input[i][1], 'go')

        title = 'Predictions of the model with :\n' + ' lr=' + \
            str(lr) + ', wd=' + str(wd) + \
            ' and mini_batch_size=' + str(mini_batch_size)
        plt.title(title)
        plt.xlabel('x coordinates')
        plt.ylabel('y coordinates')

        # reference circle
        plt.gca().add_patch(plt.Circle((0.5, 0.5), 1 /
                                       math.sqrt((2*math.pi)), color='k', alpha=0.2))
        plt.show()

    return numberOfTestErrors

# --------------------------------------- Tuning the model--------------------------------------
# This section choses different parameters value to train the model and evaluates it to keep the one with
# the best performances


def model_tuning(number_of_runs=5):
    # Fixed parameters
    nb_epochs = 100
    mini_batch_size = 50

    # Parameters to tune
    lrs = [1e-3, 5e-3, 1e-2]
    wds = [False, 1e-6]
    relu = framework.ReLu()
    tanh = framework.TanH()
    activation_functions_suggestions = [[tanh, tanh, tanh, tanh], [
        tanh, relu, relu, tanh], [relu, relu, tanh, tanh]]
    best_validation_errors = 1000

    for lr in lrs:
        for wd in wds:
            for activation_functions in activation_functions_suggestions:
                act_fun_1 = activation_functions[0]
                act_fun_2 = activation_functions[1]
                act_fun_3 = activation_functions[2]
                act_fun_4 = activation_functions[3]
                validation_errors = []
                i = 0
                while i < number_of_runs:
                    # Creation of the model
                    linear1 = framework.Linear(
                        2, 25, activation_function=act_fun_1.name)
                    linear2 = framework.Linear(
                        25, 25, activation_function=act_fun_2.name)
                    linear3 = framework.Linear(
                        25, 25, activation_function=act_fun_3.name)
                    linear4 = framework.Linear(
                        25, 1, activation_function=act_fun_4.name)

                    criterion = framework.LossMSE()
                    train_input, train_target, validation_input, validation_target = generateData(
                        1000, plotPoints=False)
                    model = framework.Sequential(
                        linear1, act_fun_1, linear2, act_fun_2, linear3, act_fun_3, linear4, act_fun_4)

                    # We compute the number of errors on the validation set and append it to a list
                    validation_error = train_model(model, criterion, train_input, train_target, validation_input, validation_target,
                                                   mini_batch_size=mini_batch_size, nb_epochs=nb_epochs, mustPrint=False, plotLoss=False, plotPoints=False)

                    # TODO: remove debug stuff
                    if validation_error < 450:
                        i += 1
                        validation_errors.append(validation_error)
                    else:
                        print(validation_error)
                        print(activation_functions)

                print('Lr =', str(lr) + ', wd =', str(wd) + ', activation functions =',
                      [act_fun_1.name, act_fun_2.name, act_fun_3.name, act_fun_4.name], ', mean validation error =', np.mean(validation_errors))

                # If the mean of the number of errors on valdiation sets is better than what we previously had, we store the current parameters
                if np.mean(validation_errors) < best_validation_errors:
                    best_validation_errors = np.mean(
                        validation_errors)
                    best_lr = lr
                    best_wd = wd
                    best_act_fun_1 = act_fun_1
                    best_act_fun_2 = act_fun_2
                    best_act_fun_3 = act_fun_3
                    best_act_fun_4 = act_fun_4

    # Printing of the best parameters
    print('--- Best parameters found ---')
    print('Lr =', str(best_lr) + ', wd =', str(wd) + ', activation functions =',
          [best_act_fun_1.name, best_act_fun_2.name, best_act_fun_3.name, best_act_fun_4.name], ', mean validation error =', best_validation_errors)
    return best_lr, best_wd, best_act_fun_1, best_act_fun_2, best_act_fun_3, best_act_fun_4

# --------------------------------------- Executing training and testing --------------------------------------
# we define here all of the layers, parameters and settings to reun the model


def test_framework(modelTuning=True):
    # Parameters
    nb_epochs = 250
    mini_batch_size = 50

    # Model tuning
    if modelTuning:
        best_lr, best_wd, best_act_fun_1, best_act_fun_2, best_act_fun_3, best_act_fun_4 = model_tuning()
    # Use directly the best parameters we found to gain some time
    else:
        relu = framework.ReLu()
        tanh = framework.TanH()
        best_lr, best_wd, best_act_fun_1, best_act_fun_2, best_act_fun_3, best_act_fun_4 = 0.01, 1e-6, tanh, relu, relu, tanh

    # Framework - hidden layers
    linear1 = framework.Linear(2, 25, activation_function=best_act_fun_1.name)
    linear2 = framework.Linear(25, 25, activation_function=best_act_fun_2.name)
    linear3 = framework.Linear(25, 25, activation_function=best_act_fun_3.name)
    linear4 = framework.Linear(25, 1, activation_function=best_act_fun_4.name)
    criterion = framework.LossMSE()

    # Generating the data
    train_input, train_target, test_input, test_target = generateData(
        1000, plotPoints=False)

    # Creating the model
    # model = framework.Sequential(linear1, relu, linear2, tanh, linear3, tanh, linear4, relu)

    # Model
    model = framework.Sequential(
        linear1, best_act_fun_1, linear2, best_act_fun_2, linear3, best_act_fun_3, linear4, best_act_fun_4)

    # Training the model
    # train_model(model, criterion, train_input, train_target, test_input, test_target, mini_batch_size=mini_batch_size, nb_epochs=nb_epochs)

    # Train model
    _ = train_model(model, criterion, train_input, train_target, test_input, test_target,
                    mini_batch_size=mini_batch_size, nb_epochs=nb_epochs, lr=best_lr, wd=best_wd, mustPrint=True, plotLoss=True, plotPoints=True)


test_framework(True)
