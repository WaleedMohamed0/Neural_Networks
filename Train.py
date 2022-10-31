
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
            net = input_feature[i][0] * weight1 + input_feature[i][1] * weight2 + bias * float(useBias)
            if net > 0:
                y = 1
            else:
                y = -1
            loss = input_feature[i][2] - y
            weight1  = weight1 + LearningRate * loss * input_feature[i][0]
            weight2 = weight2 + LearningRate * loss * input_feature[i][1] 
            bias = bias + LearningRate * loss * float(useBias)
    
    
    return weight1, weight2, bias


