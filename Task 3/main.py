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

# Train the model using backpropagation algorithm 
def train():
    global trainFeatures, trainLabels, testFeatures, testLabels, hiddenLayers, neuralsInHiddenLayer, activationFunction, learningRate, epochs, useBias
    # Assigning the activation function
    if activationFunction == "Sigmoid":
        activationFunction = sigmoid
        activationFunctionDerivative = sigmoidDerivative
    elif activationFunction == "Hyperbolic Tangent":
        activationFunction = tanh
        activationFunctionDerivative = tanhDerivative
    
    # Initializing the weights and biases
    weights = []
    biases = []
    for i in range(hiddenLayers+1):
        if i == 0:
            weights.append(np.random.rand(neuralsInHiddenLayer,len(trainFeatures[0])))
            biases.append(np.random.rand(neuralsInHiddenLayer))
        elif i == hiddenLayers:
            weights.append(np.random.rand(3,neuralsInHiddenLayer))
            biases.append(np.random.rand(3))
        else:
            weights.append(np.random.rand(neuralsInHiddenLayer, neuralsInHiddenLayer))
            biases.append(np.random.rand(neuralsInHiddenLayer))

    # Training
    for epoch in range(epochs):
        # Forward propagation
        layerOutput = []
        for i in range(trainFeatures.shape[0]):
            layerOutput.append([])
            for j in range(hiddenLayers+1):
                if j == 0:
                    layerOutput[i].append(activationFunction(np.dot(weights[j], trainFeatures[i]) + biases[j]))
                elif j == hiddenLayers:
                    layerOutput[i].append(activationFunction(np.dot(weights[j], layerOutput[i][j-1]) + biases[j]))
                else:
                    layerOutput[i].append(activationFunction(np.dot(weights[j], layerOutput[i][j-1]) + biases[j]))
                    
        print(layerOutput)
            # Backpropagation without updating the weights and biases
            
    return weights, biases

weights, biases = train()

def test():
    global trainFeatures, trainLabels, testFeatures, testLabels, hiddenLayers, neuralsInHiddenLayer, activationFunction, learningRate, epochs, useBias
    testAccuracy = 0
    for i in range(testFeatures.shape[0]):
        for j in range(hiddenLayers+1):
            if j == 0:
                layerOutput = activationFunction(np.dot(weights[j], testFeatures[i]) + biases[j])
            else:
                layerOutput = activationFunction(np.dot(weights[j], layerOutput) + biases[j])
        if np.argmax(layerOutput) == testLabels[i]:
            testAccuracy += 1
    testAccuracy /= testFeatures.shape[0]
    print(testAccuracy)
test()