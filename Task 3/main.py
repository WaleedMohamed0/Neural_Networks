from preprocessing import *
from GUI import startGUI
import numpy as np

hiddenLayers, neuralsInHiddenLayer, activationFunction, learningRate, epochs, useBias = startGUI()

hiddenLayers = int(hiddenLayers) if hiddenLayers != "" else 1
neuralsInHiddenLayer = list(map(int, neuralsInHiddenLayer)) if neuralsInHiddenLayer != "" else [5]
learningRate = float(learningRate) if learningRate != "" else 0.01
epochs = int(epochs) if epochs != "" else 100
useBias = bool(useBias) if useBias != "" else False


# Splitting the data into training and testing
Adelie = data[data['species'] == 'Adelie']
Gentoo = data[data['species'] == 'Gentoo']
Chinstrap = data[data['species'] == 'Chinstrap']


AdelieTrain, AdelieTest = Adelie[:int(0.6 * len(Adelie))], Adelie[int(0.6 * len(Adelie)):]
GentooTrain, GentooTest = Gentoo[:int(0.6 * len(Gentoo))], Gentoo[int(0.6 * len(Gentoo)):]
ChinstrapTrain, ChinstrapTest = Chinstrap[:int(0.6 * len(Chinstrap))], Chinstrap[int(0.6 * len(Chinstrap)):]

# Concatenating the data
train = pd.concat([AdelieTrain, GentooTrain, ChinstrapTrain])
test = pd.concat([AdelieTest, GentooTest, ChinstrapTest])

# Shuffling the data
train = train.sample(frac=1)
test = test.sample(frac=1)

# Splitting the data into features and labels
trainLabels = train['species']
trainSamples = train.drop(['species'], axis=1)
testLabels = test['species']
testSamples = test.drop(['species'], axis=1)

# Converting the data into numpy arrays
trainSamples = trainSamples.to_numpy()
trainLabels = trainLabels.to_numpy()
testSamples = testSamples.to_numpy()
testLabels = testLabels.to_numpy()


# Converting the labels into numerical values
def convertLabels(labels):
    for i in range(len(labels)):
        if labels[i] == "Adelie":
            labels[i] = 0
        elif labels[i] == "Gentoo":
            labels[i] = 1
        elif labels[i] == "Chinstrap":
            labels[i] = 2
    return labels


trainLabels = convertLabels(trainLabels)
testLabels = convertLabels(testLabels)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidDerivative(x):
    return x * (1 - x)


def tanh(x):
    return np.tanh(x)


def tanhDerivative(x):
    return 1 - x ** 2


def train():
    global activationFunction
    if activationFunction == "Sigmoid":
        activationFunction = sigmoid
        activationFunctionDerivative = sigmoidDerivative
    elif activationFunction == "Hyperbolic Tangent":
        activationFunction = tanh
        activationFunctionDerivative = tanhDerivative

    # initializing the weights and biases
    weights = []
    biases = []
    layerOutput = []

    # initializing the weights and biases for the rest of the hidden layers
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

    if not useBias:
        for i in range(len(biases)):
            biases[i] = np.zeros(biases[i].shape)

    for i in range(epochs):
        # forward propagation
        for j in range(len(trainSamples)):
            layerOutput.append(trainSamples[j].reshape(len(trainSamples[j]), 1))
            for k in range(hiddenLayers + 1):
                if k == 0:
                    layerOutput.append(activationFunction(
                        np.dot(weights[k], trainSamples[j].reshape(len(trainSamples[j]), 1)) + biases[k]))
                else:
                    layerOutput.append(activationFunction(np.dot(weights[k], layerOutput[k]) + biases[k]))

            # backward propagation
            # calculating the error for the output layer using the formula: error = (actual output - expected output) * derivative of activation function
            errors = []
            expectedOutput = np.zeros((3, 1))
            for k in range(3):
                if k == trainLabels[j]:
                    expectedOutput[k] = 1
                else:
                    expectedOutput[k] = 0
            errors.append((expectedOutput - layerOutput[-1]) * activationFunctionDerivative(layerOutput[-1]))
            # calculating the error for the hidden layers using the formula: error = (weights of the next layer * error of the next layer) * derivative of activation function
            for k in range(hiddenLayers):
                errors.append(np.dot(weights[-k - 1].T, errors[k]) * activationFunctionDerivative(layerOutput[-k - 2]))

            # updating the weights and biases forward step using the formula: weight = weight + learning rate * error * output of the previous layer
            for k in range(hiddenLayers + 1):
                weights[k] += learningRate * np.dot(errors[hiddenLayers - k], layerOutput[k].T)  # 2 - 0 - 2
                biases[k] += learningRate * errors[hiddenLayers - k]
            layerOutput = []
    # train accuracy
    correct = 0
    for i in range(len(trainSamples)):
        for j in range(hiddenLayers + 1):
            if j == 0:
                layerOutput = activationFunction(
                    np.dot(weights[j], trainSamples[i].reshape(len(trainSamples[i]), 1)) + biases[j])
            else:
                layerOutput = activationFunction(np.dot(weights[j], layerOutput) + biases[j])

        if np.argmax(layerOutput) == trainLabels[i]:
            correct += 1

    return weights, biases, correct / len(trainSamples)


weights, biases, trainAcc = train()
print("Train Accuracy = ", trainAcc * 100, "%")


def test(testFeatures, testLabels, weights, biases):
    global activationFunction
    if activationFunction == "Sigmoid":
        activationFunction = sigmoid
        activationFunctionDerivative = sigmoidDerivative
    elif activationFunction == "Hyperbolic Tangent":
        activationFunction = tanh
        activationFunctionDerivative = tanhDerivative

    correct = 0
    for i in range(len(testFeatures)):
        for j in range(hiddenLayers + 1):
            if j == 0:
                layerOutput = activationFunction(
                    np.dot(weights[j], testFeatures[i].reshape(len(testFeatures[i]), 1)) + biases[j])
            else:
                layerOutput = activationFunction(np.dot(weights[j], layerOutput) + biases[j])

        if np.argmax(layerOutput) == testLabels[i]:
            correct += 1
    return correct / len(testFeatures)


print("Test Accuracy = ", test(testSamples, testLabels, weights, biases) * 100, "%")
