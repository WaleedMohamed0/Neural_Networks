from msilib.schema import Feature
from tkinter import ttk
from tkinter import *
from tokenize import Special
from turtle import exitonclick
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
                
        global Special1 
        Special1 = species1.get()
        global Special2
        Special2 = species2.get() 
        species1.config(values=speciesList1)
        species2.config(values=speciesList2)
            
    def pickFeature(e):
        featureList2 = []
        featureList1 = []
        for x in featureList:
            if x != combo.get():
                featureList2.append(x)
        
        for x in featureList:
            if x != combo2.get():
                featureList1.append(x)
        global Feature1 
        Feature1 = combo.get()
        global Feature2
        Feature2 = combo2.get() 
        combo.config(values=featureList1)
        combo2.config(values=featureList2)
        
    def callback():
        global LearningRate 
        LearningRate = learningRate.get()
        global Epochs
        Epochs = epochs.get()
        global UseBias
        UseBias = useBais.get()
        
    
    
    master = Tk()
    master.title("Task 1")
    master.geometry("400x530")

    # Create Bold "Species" label
    speciesLabel = Label(master, text="Species", font=("Helvetica", 16, "bold"))
    speciesLabel.pack(pady = 5)

    # make dropdown menu
    speciesList = ["Adelie", "Chinstrap", "Gentoo"]
    speciesList1 = ["Chinstrap", "Gentoo"]


    species1 = ttk.Combobox(master, values=speciesList1)
    species1.current(0)
    species1.pack(pady=5)
    species1.bind("<<ComboboxSelected>>",pickSpecies)

    species2 = ttk.Combobox(master, values=["Adelie", "Gentoo"])
    species2.current(0)
    species2.pack(pady=20)
    species2.bind("<<ComboboxSelected>>",pickSpecies)

    # Create Bold "Feature" label
    FeatureLabel = Label(master, text="Features",font=("Helvetica", 16, "bold"))
    FeatureLabel.pack(pady = 5)

    #make dropdown menu
    featureList = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "gender","body_mass_g"]
    featureList1 = ["bill_depth_mm", "flipper_length_mm", "gender","body_mass_g"]


    combo = ttk.Combobox(master, values=featureList1)
    combo.current(0)
    combo.pack(pady = 5)
    combo.bind("<<ComboboxSelected>>",pickFeature)

    combo2 = ttk.Combobox(master, values=["bill_length_mm", "flipper_length_mm", "body_mass_g","gender"])
    combo2.current(0)
    combo2.pack(pady=20)

    combo2.bind("<<ComboboxSelected>>",pickFeature)

    # Text box for learning rate
    learningRateLabel = Label(master, text="Learning Rate",font=("Helvetica", 16, "bold"))
    learningRateLabel.pack(pady = 5)
    learningRate = Entry(master)
    learningRate.pack(pady=5)
    
    # Text Box for Epochs
    epochsLabel = Label(master, text="Epochs",font=("Helvetica", 16, "bold"))
    epochsLabel.pack(pady = 5)
    epochs = Entry(master)
    epochs.pack(pady=5)
    
    useBais = StringVar()
    ttk.Checkbutton(master,text="Use Bias", variable=useBais).pack()

    # Create "Run" with white text color button
    # if clicked, return the values and quit the GUI
    
    runButton = Button(master, text="Run", command=lambda: { 
        callback(),
        master.destroy()
                                                            })
    runButton.pack(pady = 5)
    
    #close the application if the user clicks the close button
    master.protocol("WM_DELETE_WINDOW", exit)
    
    mainloop()
    return Feature1, Feature2, Special1, Special2, LearningRate, Epochs, UseBias