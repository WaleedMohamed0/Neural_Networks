import matplotlib.pyplot as plt
import math
def visualize(feature1_1, feature1_2, feature2_1, feature2_2, weight1, weight2, bias):
    feature1_1 = [float(feature1_1[x]) for x in range(len(feature1_1))]
    feature1_2 = [float(feature1_2[x]) for x in range(len(feature1_1))]
    feature2_1 = [float(feature2_1[x]) for x in range(len(feature1_1))]
    feature2_2 = [float(feature2_2[x]) for x in range(len(feature1_1))]
    # Plot the decision boundary
    maxVal = max(max(feature1_1), max(feature2_1))
    minVal = min(min(feature1_1), min(feature2_1))
    if weight1 != 0 and weight2 != 0:
        x = [i for i in range(math.floor(minVal), math.ceil(maxVal))]
        y= [(-weight1 * i - bias) / weight2 for i in x]
        plt.plot(x, y, color='black', label='Decision Boundary')
    plt.scatter(feature1_1, feature1_2, color='red')
    plt.scatter(feature2_1, feature2_2, color='blue')
    plt.show()