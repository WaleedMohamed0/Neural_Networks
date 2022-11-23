from tkinter import ttk
from tkinter import *
from preprocessing import *


HiddenLayers = 1
NeuralsInHiddenLayer = 5
LearningRate = 0.01
Epochs = 100
UseBias = False
ActivationFunction = "Sigmoid"


def startGUI():
    def pickActivationFunction(e):
            activationFunction.config(values = ["Sigmoid", "Hyperbolic Tangent"])

    def callback():
        global LearningRate
        LearningRate = learningRate.get()
        global Epochs
        Epochs = epochs.get()
        global UseBias
        UseBias = useBias.get()
        global HiddenLayers
        HiddenLayers = hiddenLayers.get()
        global NeuralsInHiddenLayer
        NeuralsInHiddenLayer = neuralsInHiddenLayer.get()
        global ActivationFunction
        ActivationFunction = activationFunction.get()
        

    def comboCreator(values, pady, bindFunc):
        feature1_combo = ttk.Combobox(master, values=values)
        feature1_combo.current(0)
        feature1_combo.bind("<<ComboboxSelected>>", bindFunc)
        feature1_combo.pack(pady=pady)
        return feature1_combo

    def labelCreator(text):
        label = Label(master, text=text, font=("Helvetica", 16, "bold"))
        label.pack(pady=5)
        return label

    def entryCreator():
        entry = Entry(master)
        entry.pack(pady=5)
        return entry

    
    master = Tk()
    master.title("Task 1")
    master.geometry("400x570")

    numOfHiddenLayersLabel = labelCreator("Number of hidden layers")
    hiddenLayers = entryCreator()
    
    numOfNeuralsInHiddenLayerLabel = labelCreator("Number of neurals in hidden layer")
    neuralsInHiddenLayer = entryCreator()
    
    activationFunctionLabel = labelCreator("Activation function")
    activationFunction = comboCreator(["Sigmoid", "Hyperbolic Tangent"], 5, pickActivationFunction)

    learningRateLabel = labelCreator("Learning Rate")
    learningRate = entryCreator()

    epochsLabel = labelCreator("Epochs")
    epochs = entryCreator()

    useBias = StringVar()
    ttk.Checkbutton(master, text="Use Bias", variable=useBias).pack()

    # if clicked, return the values and quit the GUI
    runButton = Button(master, text="Run", command=lambda: {callback(), master.destroy()})
    runButton.pack(pady=5)

    master.protocol("WM_DELETE_WINDOW", exit)

    mainloop()

    return HiddenLayers, NeuralsInHiddenLayer,ActivationFunction , LearningRate, Epochs, UseBias
