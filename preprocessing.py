import random
import pandas as pd

data = pd.read_csv('penguins.csv')

gender = ['male', 'female']
# to replace gender data type from float to string
data = data.astype(str)

for row in range(len(data['gender'])):
    data['gender'][row] = data['gender'][row].replace("nan", random.choice(gender))

data['gender'] = data['gender'].replace({"male": 1, "female": 0})
