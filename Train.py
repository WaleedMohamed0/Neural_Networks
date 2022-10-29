import imp


import random
# Training model using signum function and perceptron learning algorithm
def train_model(feature1_1, feature1_2, feature2_1, feature2_2, label1, label2):
    weight1 = random.uniform(0,1)
    weight2 = random.uniform(0,1)
    bias = 1.0
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

