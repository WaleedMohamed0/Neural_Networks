import pandas as pd
from Train import train_model 
from Test import test_model
from preprocessing import *
from GUI import *
from Visualization import *


# get first 2 features
# feature1_1 =  data['bill_length_mm'][:30]
# feature1_2 = data['bill_depth_mm'][:30]
# feature2_1 =  data['bill_length_mm'][50:80]
# feature2_2 = data['bill_depth_mm'][50:80]
# preVis()
combo, combo2, species1,species2, learningRate, epochs, useBias = startGUI()
feature1_1 = []
feature1_2 = []
feature2_1 = []
feature2_2 = []

learningRate = float(learningRate) if learningRate != '' else 0.01
epochs = int(epochs) if epochs != '' else 100
useBias = useBias if useBias != '' else 0

# print("Feature1: "+combo)
# print("Feature2: "+combo2)
# print("species1: "+species1)
# print("species1: "+species2)
# print("learningRate: "+str(learningRate))
# print("Epochs: "+str(epochs))
# print("useBias: "+str(useBias))

for i in range(len(data)):
    if data['species'][i] == species1:
        feature1_1.append(data[combo][i])
        feature1_2.append(data[combo2][i])
        
    if data['species'][i] == species2:
        feature2_1.append(data[combo][i])
        feature2_2.append(data[combo2][i])
# convert the data to float
feature1_1 = [float(feature1_1[x]) for x in range(len(feature1_1))]
feature1_2 = [float(feature1_2[x]) for x in range(len(feature1_1))]
feature2_1 = [float(feature2_1[x]) for x in range(len(feature1_1))]
feature2_2 = [float(feature2_2[x]) for x in range(len(feature1_1))]
# divide the data to train and test
feature1_1, test1_1 = feature1_1[:30], feature1_1[30:]
feature1_2, test1_2 = feature1_2[:30], feature1_2[30:]
feature2_1, test2_1 = feature2_1[:30], feature2_1[30:]
feature2_2, test2_2 = feature2_2[:30], feature2_2[30:]



# get first 2 labels
label1 = [1.0] * 30
label2 = [-1.0] * 30

# Visualize the data
visualize(feature1_1, feature1_2, feature2_1, feature2_2, 0, 0, 0, combo,combo2, species1, species2)
# train the model
weight1, weight2, bias = train_model(feature1_1, feature1_2, feature2_1, feature2_2, label1, label2, learningRate,epochs,useBias)
# Accuracy on training data
print("Train Accuracy: ", test_model(weight1, weight2, feature1_1, feature1_2, feature2_1, feature2_2, label1, label2, bias) * 100, "%")
# visualize the decision boundary
visualize(feature1_1, feature1_2, feature2_1, feature2_2, weight1, weight2, bias,combo,combo2, species1, species2)

# test the trained model
print("Test Accuracy: Acc: ",test_model(weight1, weight2, test1_1, test1_2, test2_1, test2_2, label1, label2, bias) * 100, "%")
# visualize the decision boundary
visualize(test1_1, test1_2, test2_1, test2_2, weight1, weight2, bias,combo,combo2, species1, species2)


