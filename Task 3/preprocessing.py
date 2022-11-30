import random
import pandas as pd
from matplotlib import pyplot as plt


def preprocess(fileName = 'penguins.csv'):
    data = pd.read_csv(fileName)
    gender = ['male', 'female']
    # to replace gender data type from float to string
    data = data.astype(str)

    # replace nan in gender column with a random gender
    data['gender'] = data['gender'].replace('nan', random.choice(gender))


    data['gender'] = data['gender'].replace({"male": 1, "female": 0})

    # Normaling the data
    for i in data.columns:
        if i == 'species':
            continue
        data[i] = [float(j) for j in data[i]]
        data[i] = ((data[i] ) / (data[i].max()))
    return data


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



def splitData(data):
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
    trainLabels = convertLabels(trainLabels)
    testLabels = convertLabels(testLabels)
    return trainSamples, trainLabels, testSamples, testLabels

