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
combo, combo2, species1,species2 = startGUI()
feature1_1 = []
feature1_2 = []
feature2_1 = []
feature2_2 = []
for i in range(len(data)):
    if data['species'][i] == species1:
        feature1_1.append(data[combo][i])
        feature1_2.append(data[combo2][i])
        
    if data['species'][i] == species2:
        feature2_1.append(data[combo][i])
        feature2_2.append(data[combo2][i])

feature1_1, test1_1 = feature1_1[:30], feature1_1[30:]
feature1_2, test1_2 = feature1_2[:30], feature1_2[30:]
feature2_1, test2_1 = feature2_1[:30], feature2_1[30:]
feature2_2, test2_2 = feature2_2[:30], feature2_2[30:]


# get first 2 labels
label1 = [1.0] * 30
label2 = [-1.0] * 30


visualize(feature1_1, feature1_2, feature2_1, feature2_2, 0, 0)
weight1, weight2 = train_model(feature1_1, feature1_2, feature2_1, feature2_2, label1, label2)
visualize(feature1_1, feature1_2, feature2_1, feature2_2, weight1, weight2)


print("Acc: ",test_model(weight1, weight2, test1_1, test1_2, test2_1, test2_2, label1, label2) * 100, "%")

