import itertools
from multiprocessing.sharedctypes import Value
import random
import pandas as pd
from Train import train_model 
from Test import test_model
from matplotlib import pyplot as plt


data = pd.read_csv('penguins.csv')

gender = ['male', 'female']
# to replace gender data type from float to string
data = data.astype(str)

# fill replace nan in gender column with a random gender
data = data.replace('nan', random.choice(gender))

# print the data frame
pd.set_option('display.max_rows', None)

data['gender'] = data['gender'].replace({"male": 1, "female": 0})

# Normaling the data
for i in data.columns:
    if i == 'species':
        continue
    data[i] = [float(j) for j in data[i]]
    data[i] = ((data[i] - data[i].min()) / (data[i].max() - data[i].min()))
    
    
