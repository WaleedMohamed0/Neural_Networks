import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidDerivative(x):
    return x * (1 - x)


def tanh(x):
    return np.tanh(x)


def tanhDerivative(x):
    return 1 - x ** 2

def getActivationFunction(activationFunction):
    if activationFunction == "Sigmoid":
        return sigmoid, sigmoidDerivative
    if activationFunction == "Hyperbolic Tangent":
        return tanh, tanhDerivative

def train(hiddenLayers, neuralsInHiddenLayer, activationFunction, learningRate, epochs, useBias, trainSamples, trainLabels):

    activationFunction, activationFunctionDerivative = getActivationFunction(activationFunction)

    # initializing the weights and biases
    weights = []
    biases = []
    layerOutput = []

    # initializing the weights and biases
    weights, biases = init(hiddenLayers, neuralsInHiddenLayer, trainSamples)

    if not useBias:
        for i in range(len(biases)):
            biases[i] = np.zeros(biases[i].shape)

    for i in range(epochs):
        for j in range(len(trainSamples)):
            
            layerOutput = fowrardPropagation(weights, biases, trainSamples[j], activationFunction, hiddenLayers)
            
            errors = backPropagation(weights, trainLabels[j], activationFunctionDerivative, hiddenLayers, layerOutput)
            
            weights, biases = updateWeights(weights, biases, learningRate, errors, layerOutput, hiddenLayers)
            
    # train accuracy
    trainAccuracy = accuracy(weights, biases, trainSamples, trainLabels, activationFunction, hiddenLayers)

    return weights, biases, trainAccuracy


def init(hiddenLayers, neuralsInHiddenLayer, trainSamples):
    weights = []
    biases = []
    for i in range(hiddenLayers):
        if i == 0:
            weights.append(np.random.uniform(0, 1, (neuralsInHiddenLayer[i], len(trainSamples[0]))))
            biases.append(np.random.uniform(0, 1, (neuralsInHiddenLayer[i], 1)))
        else:
            weights.append(np.random.uniform(0, 1, (neuralsInHiddenLayer[i], neuralsInHiddenLayer[i - 1])))
            biases.append(np.random.uniform(0, 1, (neuralsInHiddenLayer[i], 1)))
    # initializing the weights and biases for the output layer
    if hiddenLayers == 0:
        weights.append(np.random.uniform(0, 1, (3, 5)))
        biases.append(np.random.uniform(0, 1, (3, 1)))
    else:
        weights.append(np.random.uniform(0, 1, (3, neuralsInHiddenLayer[-1])))
        biases.append(np.random.uniform(0, 1, (3, 1)))
            
    return weights, biases


def fowrardPropagation(weights, biases, trainSample, activationFunction, hiddenLayers):
    layerOutput = []
    layerOutput.append(trainSample.reshape(len(trainSample), 1))
    for k in range(hiddenLayers + 1):
        if k == 0:
            layerOutput.append(activationFunction(
                np.dot(weights[k], trainSample.reshape(len(trainSample), 1)) + biases[k]))
        else:
            layerOutput.append(activationFunction(np.dot(weights[k], layerOutput[k]) + biases[k]))
    return layerOutput


def backPropagation(weights, trainLabel, activationFunctionDerivative, hiddenLayers, layerOutput):
    errors = []
    expectedOutput = np.zeros((3, 1))
    for k in range(3):
        if k == trainLabel:
            expectedOutput[k] = 1
        else:
            expectedOutput[k] = 0
    errors.append((expectedOutput - layerOutput[-1]) * activationFunctionDerivative(layerOutput[-1]))
    # calculating the error for the hidden layers using the formula: error = (weights of the next layer * error of the next layer) * derivative of activation function
    for k in range(hiddenLayers):
        errors.append(np.dot(weights[-k - 1].T, errors[k]) * activationFunctionDerivative(layerOutput[-k - 2]))
    return errors


def updateWeights(weights, biases, learningRate, errors, layerOutput, hiddenLayers):
    for k in range(hiddenLayers + 1):
        weights[k] += learningRate * np.dot(errors[hiddenLayers - k], layerOutput[k].T)  # 2 - 0 - 2
        biases[k] += learningRate * errors[hiddenLayers - k]
    return weights, biases


def accuracy(weights, biases, Samples, Labels, activationFunction, hiddenLayers):
    if activationFunction != sigmoid and activationFunction != tanh:
        activationFunction, activationFunctionDerivative = getActivationFunction(activationFunction)
    correct = 0
    for i in range(len(Samples)):
        layerOutput = fowrardPropagation(weights, biases, Samples[i], activationFunction, hiddenLayers)
        if np.argmax(layerOutput[-1]) == Labels[i]:
            correct += 1
            
    return correct / len(Samples)

def visualizeConfusionMatrix(confusionMatrix):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusionMatrix)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    for (i, j), z in np.ndenumerate(confusionMatrix):
        ax.text(j, i, z, ha='center', va='center')
    tick_marks = np.arange(3)
    plt.xticks(tick_marks,['Adelie', 'Gentoo', 'Chinstrap'])
    plt.yticks(tick_marks,['Adelie', 'Gentoo', 'Chinstrap'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def confusionMatrix(weights, biases, Samples, Labels, activationFunction, hiddenLayers):
    if activationFunction != sigmoid and activationFunction != tanh:
        activationFunction, activationFunctionDerivative = getActivationFunction(activationFunction)
    confusionMatrix = np.zeros((3, 3))
    for i in range(len(Samples)):
        layerOutput = fowrardPropagation(weights, biases, Samples[i], activationFunction, hiddenLayers)
        confusionMatrix[Labels[i]][np.argmax(layerOutput[-1])] += 1
    print(confusionMatrix)
    visualizeConfusionMatrix(confusionMatrix)
    return confusionMatrix