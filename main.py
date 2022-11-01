import pandas as pd
from Train import train_model 
from Test import test_model
from preprocessing import *
from GUI import *
from Visualization import *



preVis()
feature1, feature2, species1, species2, learningRate, epochs, useBias = startGUI()
# Feature1_1 is the feature1 of species1
# Feature1_2 is the feature2 of species1
# Feature2_1 is the feature1 of species2
# Feature2_2 is the feature2 of species2
feature1_1 = []
feature1_2 = []
feature2_1 = []
feature2_2 = []

learningRate = float(learningRate) if learningRate != '' else 0.01
epochs = int(epochs) if epochs != '' else 100
useBias = useBias if useBias != '' else 0

# print("Feature1: "+feature1)
# print("feature2: "+feature2)
# print("species1: "+species1)
# print("species1: "+species2)
# print("learningRate: "+str(learningRate))
# print("Epochs: "+str(epochs))
# print("useBias: "+str(useBias))

for i in range(len(data)):
    if data['species'][i] == species1:
        feature1_1.append(data[feature1][i])
        feature1_2.append(data[feature2][i])
        
    if data['species'][i] == species2:
        feature2_1.append(data[feature1][i])
        feature2_2.append(data[feature2][i])
# convert the data to float
feature1_1 = [float(x) for x in feature1_1]
feature1_2 = [float(x) for x in feature1_2]
feature2_1 = [float(x) for x in feature2_1]
feature2_2 = [float(x) for x in feature2_2]
# divide the data to train and test
feature1_1, test1_1 = feature1_1[:30], feature1_1[30:]
feature1_2, test1_2 = feature1_2[:30], feature1_2[30:]
feature2_1, test2_1 = feature2_1[:30], feature2_1[30:]
feature2_2, test2_2 = feature2_2[:30], feature2_2[30:]


# create first 2 labels
label1 = [1.0] * 30
label2 = [-1.0] * 30

# Visualize the data
visualize(feature1_1, feature1_2, feature2_1, feature2_2, 0, 0, 0, feature1,feature2, species1, species2)

# train the model
weight1, weight2, bias = train_model(feature1_1, feature1_2, feature2_1, feature2_2, label1, label2, learningRate,epochs,useBias)

# Accuracy on training data
print("Train Accuracy: ", test_model(weight1, weight2, feature1_1, feature1_2, feature2_1, feature2_2, label1, label2, bias) * 100, "%")

# visualize the decision boundary
visualize(feature1_1, feature1_2, feature2_1, feature2_2, weight1, weight2, bias,feature1,feature2, species1, species2)

# test the trained model
print("Test Accuracy: Acc: ",test_model(weight1, weight2, test1_1, test1_2, test2_1, test2_2, label1, label2, bias) * 100, "%")
# visualize the decision boundary
visualize(test1_1, test1_2, test2_1, test2_2, weight1, weight2, bias,feature1,feature2, species1, species2)

confusion_matrix(weight1, weight2, test1_1, test1_2, test2_1, test2_2,species1,species2, label1, label2, bias)
