

import random
# Training model using signum function and perceptron learning algorithm
def train_model(feature1_1, feature1_2, feature2_1, feature2_2, label1, label2, LearningRate, Epochs, useBias):
    weight1 = random.uniform(0,1)
    weight2 = random.uniform(0,1)
    bias = random.uniform(0,1) if useBias else 0
    input_feature = list(zip(feature1_1, feature1_2, label1)) + list(zip(feature2_1, feature2_2, label2))
    random.shuffle(input_feature)
    for epoch in range(Epochs):
        for i in range(len(input_feature)):
            feature0 = float(input_feature[i][0])
            feature1 = float(input_feature[i][1])
            feature2 = float(input_feature[i][2])
            net = feature0 * weight1 + feature1 * weight2 + bias * float(useBias)
            if net > 0:
                y = 1
            else:
                y = -1
            loss = feature2 - y
            weight1  = weight1 + LearningRate * loss * feature0
            weight2 = weight2 + LearningRate * loss * feature1 
            bias = bias + LearningRate * loss * float(useBias)
    
    
    return weight1, weight2, bias


