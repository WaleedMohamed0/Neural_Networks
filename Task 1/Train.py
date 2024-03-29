
import random
# Training model using signum function and perceptron learning algorithm
def train_model(feature1_1, feature1_2, feature2_1, feature2_2, label1, label2, LearningRate, Epochs, useBias):
    weight1 = random.uniform(0,1)
    weight2 = random.uniform(0,1)
    bias = random.uniform(0,1) if useBias else 0
    input_feature = list(zip(feature1_1, feature1_2, label1)) + list(zip(feature2_1, feature2_2, label2))
    # Gentoo , Chenstrap
    random.shuffle(input_feature)
    for epoch in range(Epochs):
        for i in range(len(input_feature)):
            net = calculate_net(weight1, weight2, bias, input_feature[i][0], input_feature[i][1])
            y = signum(net)
            loss = input_feature[i][2] - y
            weight1, weight2, bias = update_weights(weight1, weight2, bias, LearningRate, loss, input_feature[i][0], input_feature[i][1], useBias)
    
    
    return weight1, weight2, bias

def calculate_net(weight1, weight2, bias, feature1, feature2):
    return weight1 * feature1 + weight2 * feature2 + bias


def update_weights(weight1, weight2, bias, LearningRate, loss, feature1, feature2, useBias):
    weight1 = weight1 + LearningRate * loss * feature1
    weight2 = weight2 + LearningRate * loss * feature2
    bias = bias + LearningRate * loss * float(useBias)
    return weight1, weight2, bias

# Activation function
def signum(x):
    if x > 0:
        return 1
    else:
        return -1


