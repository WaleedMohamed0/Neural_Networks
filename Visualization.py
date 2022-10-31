import matplotlib.pyplot as plt
from preprocessing import *

def visualize(feature1_1, feature1_2, feature2_1, feature2_2, weight1, weight2, bias, feature1, feature2, species1, species2):
    # Plot the decision boundary
    plt.scatter(feature1_1, feature1_2, color='red', label=species1)
    plt.scatter(feature2_1, feature2_2, color='blue', label=species2)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    if weight1 != 0 and weight2 != 0:
        minX = [min(feature1_1), min(feature2_1)]
        maxX = [max(feature1_1), max(feature2_1)]
        x = [min(minX),max(maxX)]
        y= [(-weight1 * i - bias) / weight2 for i in x]
        plt.plot(x, y, color='black', label='Decision Boundary')
        
    plt.legend([species1, species2])

    plt.show()

# Define a Visualization function to visualize all the data points before train
def preVis():
    AdelieFeature1 = []
    AdelieFeature2 = []

    GentooFeature1 = []
    GentooFeature2 = []
    
    ChinstrapFeature1 = []
    ChinstrapFeature2 = []
    
    featureList = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "gender","body_mass_g"]
    # a for loop to get all 10 combinations of features with species
    for i in range(len(featureList) - 1):
        for j in range(i + 1, len(featureList)):
            # a for loop to get all 3 species
            for k in range(len(data)):
                if data['species'][k] == "Adelie":
                    AdelieFeature1.append(float(data[featureList[i]][k]))
                    AdelieFeature2.append(float(data[featureList[j]][k]))
                if data['species'][k] == "Gentoo":
                    GentooFeature1.append(float(data[featureList[i]][k]))
                    GentooFeature2.append(float(data[featureList[j]][k]))
                if data['species'][k] == "Chinstrap":
                    ChinstrapFeature1.append(float(data[featureList[i]][k]))
                    ChinstrapFeature2.append(float(data[featureList[j]][k]))
            # plot the data
            plt.scatter(AdelieFeature1, AdelieFeature2, color='red')
            plt.scatter(GentooFeature1, GentooFeature2, color='blue')
            plt.scatter(ChinstrapFeature1, ChinstrapFeature2, color='green')
            plt.legend(['Adelie', 'Gentoo', 'Chinstrap'])
            plt.xlabel(featureList[i])
            plt.ylabel(featureList[j])
            plt.show()
            # clear the data for next plot
            AdelieFeature1 = []
            AdelieFeature2 = []
            GentooFeature1 = []
            GentooFeature2 = []
            ChinstrapFeature1 = []
            ChinstrapFeature2 = []