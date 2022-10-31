from tkinter import ttk
from tkinter import *
from preprocessing import *

Feature1 = "bill_depth_mm"
Feature2 = "bill_length_mm"
Special1 = "Chinstrap"
Special2 = "Adelie"
LearningRate = 0.01
Epochs = 100
UseBias = False

def startGUI():
    def pickSpecies(e):
        speciesList2 = []
        speciesList1 = []
        for x in speciesList:
            if x != species1.get():
                speciesList2.append(x)
        
        for x in speciesList:
            if x != species2.get():
                speciesList1.append(x)

        species1.config(values=speciesList1)
        species2.config(values=speciesList2)
            
    def pickFeature(e):
        featureList2 = []
        featureList1 = []
        for x in featureList:
            if x != feature1_combo.get():
                featureList2.append(x)
        
        for x in featureList:
            if x != feature2_combo.get():
                featureList1.append(x)

        feature1_combo.config(values=featureList1)
        feature2_combo.config(values=featureList2)
        
    def callback():
        global LearningRate 
        LearningRate = learningRate.get()
        global Epochs
        Epochs = epochs.get()
        global UseBias
        UseBias = useBais.get()
        global Feature1
        Feature1 = feature1_combo.get()
        global Feature2
        Feature2 = feature2_combo.get()
        global Special1
        Special1 = species1.get()
        global Special2
        Special2 = species2.get()
        
        
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
    
    speciesList = ["Adelie", "Chinstrap", "Gentoo"]
    speciesList1 = ["Chinstrap", "Gentoo"]
    featureList = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "gender","body_mass_g"]
    featureList1 = ["bill_depth_mm", "flipper_length_mm", "gender","body_mass_g"]
    
    master = Tk()
    master.title("Task 1")
    master.geometry("400x530")

    speciesLabel = labelCreator("Species")
    species1 = comboCreator(speciesList1, 5, pickSpecies)
    species2 = comboCreator(["Adelie", "Gentoo"], 20, pickSpecies)

    FeatureLabel = labelCreator("Feature")
    feature1_combo = comboCreator(featureList1, 5, pickFeature)
    feature2_combo = comboCreator(["bill_length_mm", "flipper_length_mm", "body_mass_g","gender"], 20, pickFeature)
    
    learningRateLabel = labelCreator("Learning Rate")
    learningRate = entryCreator()
    
    epochsLabel = labelCreator("Epochs")
    epochs = entryCreator()
    
    useBais = StringVar()
    ttk.Checkbutton(master,text="Use Bias", variable=useBais).pack()

    # Create "Run" with white text color button
    # if clicked, return the values and quit the GUI
    
    runButton = Button(master, text="Run", command=lambda: { callback(), master.destroy() })
    runButton.pack(pady = 5)
    
    master.protocol("WM_DELETE_WINDOW", exit)
    
    mainloop()
    return Feature1, Feature2, Special1, Special2, LearningRate, Epochs, UseBias