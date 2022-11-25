from preprocessing import *
from GUI import startGUI
import numpy as np

hiddenLayers, neuralsInHiddenLayer,activationFunction, learningRate, epochs, useBias = startGUI()

hiddenLayers = int(hiddenLayers) if hiddenLayers != "" else 1
neuralsInHiddenLayer = int(neuralsInHiddenLayer) if neuralsInHiddenLayer != "" else 5
learningRate = float(learningRate) if learningRate != "" else 0.01
epochs = int(epochs) if epochs != "" else 100
useBias = bool(useBias) if useBias != "" else False

# Splitting the data into training and testing
Adelie = data[data['species'] == 'Adelie']
Gentoo = data[data['species'] == 'Gentoo']
Chinstrap = data[data['species'] == 'Chinstrap']

# Adelie = Adelie.sample(frac = 1)
# Gentoo = Gentoo.sample(frac = 1)
# Chinstrap = Chinstrap.sample(frac = 1)

AdelieTrain,AdelieTest = Adelie[:int(0.6*len(Adelie))], Adelie[int(0.6*len(Adelie)):]
GentooTrain,GentooTest = Gentoo[:int(0.6*len(Gentoo))], Gentoo[int(0.6*len(Gentoo)):]
ChinstrapTrain,ChinstrapTest = Chinstrap[:int(0.6*len(Chinstrap))], Chinstrap[int(0.6*len(Chinstrap)):]

# Concatenating the data
train = pd.concat([AdelieTrain, GentooTrain, ChinstrapTrain])
test = pd.concat([AdelieTest, GentooTest, ChinstrapTest])

# Shuffling the data
train = train.sample(frac = 1)
test = test.sample(frac = 1)


# Splitting the data into features and labels
trainLabels = train['species']
trainFeatures = train.drop(['species'], axis = 1)
testLabels = test['species']
testFeatures = test.drop(['species'], axis = 1)

# Converting the data into numpy arrays
trainFeatures = trainFeatures.to_numpy()
trainLabels = trainLabels.to_numpy()
testFeatures = testFeatures.to_numpy()
testLabels = testLabels.to_numpy()

# Converting the labels into numirical values
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
    return sigmoid(x) * (1 - sigmoid(x))
def tanh(x):
    return np.tanh(x)
def tanhDerivative(x):
    return 1 - np.tanh(x) ** 2

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
    
    # initializing the weights and biases for the first hidden layer
    weights.append(np.random.rand(neuralsInHiddenLayer, len(trainFeatures[0])))
    biases.append(np.random.rand(neuralsInHiddenLayer, 1)) 
    # initializing the weights and biases for the rest of the hidden layers
    for i in range(hiddenLayers - 1):
        weights.append(np.random.rand(neuralsInHiddenLayer, neuralsInHiddenLayer))
        biases.append(np.random.rand(neuralsInHiddenLayer, 1))
        
    # initializing the weights and biases for the output layer
    weights.append(np.random.rand(3, neuralsInHiddenLayer))
    biases.append(np.random.rand(3, 1))
    
    for i in range(epochs):
        # forward propagation
        for j in range(len(trainFeatures)):
            for k in range(hiddenLayers+1):
                if k == 0:
                    layerOutput.append(activationFunction(np.dot(weights[k], trainFeatures[j].reshape(len(trainFeatures[j]), 1)) + biases[k]))
                else:
                    layerOutput.append(activationFunction(np.dot(weights[k], layerOutput[k - 1]) + biases[k]))
                    
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
                errors.append(np.dot(weights[-k-1].T, errors[k]) * activationFunctionDerivative(layerOutput[-k-2]))
            
            # updating the weights and biases forward step using the formula: weight = weight + learning rate * error * output of the previous layer
            for k in range(hiddenLayers+1):
                weights[k] += learningRate * np.dot(errors[hiddenLayers-k].T, layerOutput[hiddenLayers-k-1])
                biases[k] += learningRate * errors[hiddenLayers-k]
            
            layerOutput = []
            
            
    return weights, biases

weights, biases=train()

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
        for j in range(hiddenLayers+1):
            if j == 0:
                layerOutput = activationFunction(np.dot(weights[j], testFeatures[i].reshape(len(testFeatures[i]), 1)) + biases[j])
            else:
                layerOutput = activationFunction(np.dot(weights[j], layerOutput) + biases[j])
        print(np.argmax(layerOutput))

        if np.argmax(layerOutput) == testLabels[i]:
            correct += 1
    return correct/len(testFeatures)
print(test(testFeatures, testLabels, weights, biases))