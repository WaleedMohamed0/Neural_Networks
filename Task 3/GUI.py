from tkinter import ttk, messagebox
from tkinter import *
from preprocessing import *

HiddenLayers = 1
NeuralsInHiddenLayer = [5]
LearningRate = 0.01
Epochs = 100
UseBias = False
ActivationFunction = "Sigmoid"


def startGUI():
    def pickActivationFunction(e):
        activationFunction.config(values=["Sigmoid", "Hyperbolic Tangent"])

    def callback():
        global LearningRate
        LearningRate = learningRate.get()
        global Epochs
        Epochs = epochs.get()
        global UseBias
        UseBias = useBias.get()
        global HiddenLayers
        HiddenLayers = hiddenLayers.get()
        if HiddenLayers == "":
            HiddenLayers = 1
        global NeuralsInHiddenLayer
        NeuralsInHiddenLayer = neuralsInHiddenLayer.get()
        if NeuralsInHiddenLayer == "":
            NeuralsInHiddenLayer = [5]

        global ActivationFunction
        ActivationFunction = activationFunction.get()

    def checkNumOfHiddens():
        global NeuralsInHiddenLayer

        if ',' in NeuralsInHiddenLayer:
            NeuralsInHiddenLayer = NeuralsInHiddenLayer.split(",")
            NeuralsInHiddenLayer = [int(i) for i in NeuralsInHiddenLayer]

        if len(NeuralsInHiddenLayer) != int(HiddenLayers):
            messagebox.showerror("Error",
                                 "Number of hidden layers and number of neurons in hidden layers are not equal")
            return False
        return True

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
    # to add default value
    hiddenLayers.insert(0, int(HiddenLayers))

    numOfNeuralsInHiddenLayerLabel = labelCreator("Number of neurals in hidden layer")
    neuralsInHiddenLayer = entryCreator()
    neuralsInHiddenLayer.insert(0, int(NeuralsInHiddenLayer[0]))

    activationFunctionLabel = labelCreator("Activation function")
    activationFunction = comboCreator(["Sigmoid", "Hyperbolic Tangent"], 5, pickActivationFunction)

    learningRateLabel = labelCreator("Learning Rate")
    learningRate = entryCreator()
    learningRate.insert(0, float(LearningRate))
    
    epochsLabel = labelCreator("Epochs")
    epochs = entryCreator()
    epochs.insert(0, int(Epochs))

    useBias = StringVar()
    ttk.Checkbutton(master, text="Use Bias", variable=useBias).pack()

    # if clicked, return the values and quit the GUI
    runButton = Button(master, text="Run", width=8, bg="white", font=("Helvetica", 15, "bold"),
                       command=lambda: {callback(), master.destroy() if checkNumOfHiddens() else None})
    runButton.pack(pady=5)

    master.protocol("WM_DELETE_WINDOW", exit)

    mainloop()

    return HiddenLayers, NeuralsInHiddenLayer, ActivationFunction, LearningRate, Epochs, UseBias
