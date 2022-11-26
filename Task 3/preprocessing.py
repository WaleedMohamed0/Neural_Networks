import random
import pandas as pd
from matplotlib import pyplot as plt


data = pd.read_csv('penguins.csv')

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
    data[i] = ((data[i] ) / (data[i].max() ))
