import random
import matplotlib.pyplot as plt


# Training model using signum function and perceptron learning algorithm
def train_model(feature1_1, feature1_2, feature2_1, feature2_2, label1, label2, LearningRate, Epochs, useBias, minMSE):
    errors = []
    weight1 = random.uniform(0, 1)
    weight2 = random.uniform(0, 1)
    bias = random.uniform(0, 1) if useBias else 0
    visualizationList = []
    input_feature = list(zip(feature1_1, feature1_2, label1)) + list(zip(feature2_1, feature2_2, label2))
    random.shuffle(input_feature)
    for epoch in range(Epochs):
        for i in range(len(input_feature)):
            net = calculate_net(weight1, weight2, bias, input_feature[i][0], input_feature[i][1])
            loss = input_feature[i][2] - net
            weight1, weight2, bias = update_weights(weight1, weight2, bias, LearningRate, loss, input_feature[i][0],
                                                    input_feature[i][1], useBias)
        for i in range(len(input_feature)):
            net = calculate_net(weight1, weight2, bias, input_feature[i][0], input_feature[i][1])
            loss = input_feature[i][2] - net
            errors.append(loss)
        visualizationList.append(0.5 * (sum([i ** 2 for i in errors]) / len(errors)))
        # If MSE is 0, stop training
        # 1/2 * (sum(errors) ** 2 / len(errors))
        if 0.5 * (sum([i ** 2 for i in errors]) / len(errors)) <= minMSE:
            break

    plt.plot(visualizationList)
    plt.show()
    return weight1, weight2, bias, errors


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
