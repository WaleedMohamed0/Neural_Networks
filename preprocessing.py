from cProfile import label
import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


data = pd.read_csv('penguins.csv')

gender = ['male', 'female']
# to replace gender data type from float to string
data = data.astype(str)

for row in range(len(data['gender'])):
    data['gender'][row] = data['gender'][row].replace("nan", random.choice(gender))

data['gender'] = data['gender'].replace({"male": 1, "female": 0})

# get first 2 features
feature1_1 =  data['bill_length_mm'][:30]
feature1_2 = data['bill_depth_mm'][:30]
feature2_1 =  data['bill_length_mm'][50:80]
feature2_2 = data['bill_depth_mm'][50:80]


# get first 2 labels
label1 = [1.0] * 30
label2 = [-1.0] * 30

# Training model using signum function and perceptron learning algorithm
def train_model(feature1_1, feature1_2, feature2_1, feature2_2, label1, label2):
    weight1 = random.uniform(0,1)
    weight2 = random.uniform(0,1)
    bias = 0.0
    learning_rate = 0.01
    epochs = 100
    input_feature = list(zip(feature1_1, feature1_2, label1)) + list(zip(feature2_1, feature2_2, label2))
    random.shuffle(input_feature)
    for epoch in range(epochs):
        for i in range(len(input_feature)):
            feature0 = float(input_feature[i][0])
            feature1 = float(input_feature[i][1])
            feature2 = float(input_feature[i][2])
            
            net = feature0 * weight1 + feature1 * weight2 + bias
            if net > 0:
                y = 1
            else:
                y = -1
            loss = feature2 - y
            weight1  = weight1 + learning_rate * loss * feature0
            weight2 = weight2 + learning_rate * loss * feature1
        
    return weight1, weight2
    
weight1, weight2 = train_model(feature1_1, feature1_2, feature2_1, feature2_2, label1, label2)

def test_model(weight1, weight2, feature1_1, feature1_2, feature2_1, feature2_2, label1, label2):
	correct = 0
	input_feature = list(zip(feature1_1, feature1_2, label1)) + list(zip(feature2_1, feature2_2, label2))
	for i in range(len(input_feature)):
		feature0 = float(input_feature[i][0])
		feature1 = float(input_feature[i][1])
		feature2 = float(input_feature[i][2])
		net = feature0 * weight1 + feature1 * weight2
		if net > 0:
			y = 1
		else:
			y = -1
		if y == feature2:
			correct += 1
	return correct/len(input_feature)

test1_1 =  data['bill_length_mm'][30:50]
test1_2 = data['bill_depth_mm'][30:50]
test2_1 =  data['bill_length_mm'][80:100]
test2_2 = data['bill_depth_mm'][80:100]
print("Acc: ",test_model(weight1, weight2, test1_1, test1_2, test2_1, test2_2, label1, label2) * 100, "%")








# # visualize the data and the decision boundary
# def visualize_data(feature1_1, feature1_2, feature2_1, feature2_2, weight1, weight2):
#     finalData = list(zip(label1,feature1_1, feature1_2,label1)) + list(zip(label1,feature2_1, feature2_2,label2))
#     print(finalData)
# visualize_data(feature1_1, feature1_2, feature2_1, feature2_2, weight1, weight2)


# weight1 = 0.0
#     weight2 = 0.0
#     bias = 0.0
#     learning_rate = 0.01
#     epochs = 100
#     # combine 2 features and labels
#     input_feature = list(zip(feature1_1, feature1_2, label1)) + list(zip(feature2_1, feature2_2, label2))
#     for epoch in range(epochs):
#         random.shuffle(input_feature)
#         for feature, feature2, label in input_feature:
#             prediction = weight1 * feature + weight2 * feature2 + bias
#             if prediction > 0:
#                 prediction = 1
#             else:
#                 prediction = -1
#             weight1 += learning_rate * (label - prediction) * feature
#             weight2 += learning_rate * (label - prediction) * feature2
#             bias += learning_rate * (label - prediction)
#     return weight1, weight2, bias
# AdelieClass = data['species'].loc[:49].replace("Adelie", -1)
# GentooClass = data['species'].loc[50:99].replace("Gentoo", 1)

# Adelie_train = AdelieClass[:30]
# Gentoo_train = GentooClass[:30]
# rand_W = random.uniform(0, 1)

# train the model using Signum function and single perceptron