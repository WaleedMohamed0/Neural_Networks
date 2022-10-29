import matplotlib.pyplot as plt
def visualize(feature1_1, feature1_2, feature2_1, feature2_2, weight1, weight2):
    # Plot the decision boundary
    # w1x + w2y  = 0
    # y = -w1/w2 * x
    maxVal = max(float(max(feature1_1)), float(max(feature2_1)))
    print(maxVal)
    if weight1 != 0 and weight2 != 0:
        x = [i for i in range(0, int(maxVal))]
        y = [-(weight1 * i) / weight2 for i in x]
        plt.plot(x, y, color='black', label='Decision Boundary')
    plt.scatter(feature1_1, feature1_2, color='red')
    plt.scatter(feature2_1, feature2_2, color='blue')
    plt.show()