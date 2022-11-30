from preprocessing import *
from GUI import startGUI
import numpy as np
from model import *

hiddenLayers, neuralsInHiddenLayer, activationFunction, learningRate, epochs, useBias = startGUI()

hiddenLayers = int(hiddenLayers) if hiddenLayers != "" else 1
neuralsInHiddenLayer = list(map(int, neuralsInHiddenLayer)) if neuralsInHiddenLayer != "" else [5]
learningRate = float(learningRate) if learningRate != "" else 0.01
epochs = int(epochs) if epochs != "" else 100
useBias = bool(useBias) if useBias != "" else False

data = preprocess('penguins.csv')

trainSamples, trainLabels, testSamples, testLabels = splitData(data)


weights, biases, trainAcc = train(hiddenLayers, neuralsInHiddenLayer, activationFunction, learningRate, epochs, useBias, trainSamples, trainLabels)
print("Train Accuracy = ", trainAcc * 100, "%")

print("Test Accuracy = ", accuracy(weights,biases,testSamples,testLabels,activationFunction,hiddenLayers) * 100, "%")
confusionMatrix(weights,biases,testSamples,testLabels,activationFunction,hiddenLayers)
