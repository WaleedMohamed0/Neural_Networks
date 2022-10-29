import itertools
import random
import pandas as pd
import numpy as np
from Train import train_model 
from Test import test_model
from matplotlib import pyplot as plt


data = pd.read_csv('penguins.csv')

gender = ['male', 'female']
# to replace gender data type from float to string
data = data.astype(str)

for row in range(len(data['gender'])):
    data['gender'][row] = data['gender'][row].replace("nan", random.choice(gender))

data['gender'] = data['gender'].replace({"male": 1, "female": 0})







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