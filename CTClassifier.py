from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import logging

class COVIDCT_Classifier_GUI:

    def __init__(self):
        logging.getLogger("tensorflow").setLevel(logging.WARNING)

        #load trained models
        Vgg16 = load_model("Vgg16")
        EfficientNetB5 = load_model("EfficientNetB5")
        #store trained models in a list
        self.base_classifiers = [Vgg16,EfficientNetB5]

        #create image path class variable
        self.image_path = ""

        #create the graphical window pane
        self.window = Tk()
        self.window.title("COVID-19 CT-scan Classifier")
        self.window.geometry("500x500")


        self.btn = Button(self.window, text="Select CT-scan", fg="blue", command=self.get_image, borderwidth=5,
                     relief="raised")
        self.btn.place(x=200,y=95)

        self.image_container = Label(self.window, image=None)
        self.image_container.place(x = 150, y = 200)

        self.diagnosis = Label(self.window)
        self.diagnosis.place(x=170,y=150)

        self.window.mainloop()





    #open file dialog to select an image
    def get_image(self):
     file_name = askopenfilename()
     # load and store image
     img = Image.open(file_name)
     resized_img = img.resize((200,200),Image.ANTIALIAS)
     CT_image = ImageTk.PhotoImage(resized_img)

     # place image in label
     self.image_container.configure(image=CT_image)
     self.image_container.image = CT_image



     # predict the diagnosis of the selected CT scan
     self.predict_CT(file_name)


    def predict_CT(self,CT_scan):
        # read and convert image to image tensor for input into keras models
        img = image.load_img(CT_scan, target_size=(224,224,3))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)


        # combine predictions of base classifiers with soft voting
        average_votes = [0] * len(self.base_classifiers)
        for bc in self.base_classifiers:
            predictions = bc.predict(img)
            i = 0
            for p in predictions[0]:
                average_votes[i] = average_votes[i] + p
                i = i + 1

        for j in range(len(average_votes)):
            average_votes[j] = average_votes[j]/len(self.base_classifiers)


        # display covid-19 diagnosis on the GUI
        if(average_votes[0]>average_votes[1]):
            self.diagnosis.configure(text="Diagnosis: COVID-19 Positive",fg="red")
        else:
            self.diagnosis.configure(text="Diagnosis: COVID-19 Negative",fg="green")



if __name__=="__main__":
    COVIDCT_Classifier_GUI()
