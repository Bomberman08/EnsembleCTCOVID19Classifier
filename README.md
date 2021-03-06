# EnsembleCTCOVID19Classifier
Note: you will need to have a python interpreter installed on your computer to run this software. This software makes use
      of pre-trained Vgg16 and EfficientNetB5 transfer learning models to form hard and soft voting ensemble classifiers.
      The final classifier software makes use of a graphical interface and only uses the soft voting ensemble classifier to
      classify CT scans in real time.

Step 1: Download and unzip the source code files from Github using the link provided in the "Github.txt file".

Step 2: Open the "TrainedModelsLink.txt" text file, copy and paste the link on your web browser and hit search.
        You will be directed to a google drive page where there are 2 folders called "Vgg16" and "EfficientNetB5."
        Download both folders and unzip them into the directory where you saved the source code files. Note that both
        folders, "Vgg16" and "EfficientNetB5", must be unzipped and in the exact same directory as the source code files.      

Step 3: Download the dataset using the link provided in the "LinkToDataset.txt" file. 
        Unzip the dataset.

Step 4: Move the downloaded dataset folder to the directory that you saved the source code files.
        Rename the dataset folder to "Dataset"

Step 5: Open a python IDE or a terminal to run the source code files.
        To run on command line, control directory or cd to the directory of the source code files.
        Type python then press space bar and type in the name of the python file you want to run.

Step 6: Run the source code file "Prelim.py" which will install the relavent libraries needed to run the code
        and will also randomly split the dataset into "train", "val" and "test folders". These will be
        stored within a new folder called "SplitDataset". Make sure to wait until
        the code has completed running and ensure you have internet connection. 

Step 7: You have 3 files you can run from here. See step 8 for running "CTClassifier.py", see step 9 for running "EnsembleTester.py",
        and see step 10 for running "EnsembleTrainer.py". 


***Note that you may see Warnings in your terminal when running each of the 3 python files.
This does not mean there are errors in the software ; these are simply warnings automatically displayed by the tensorflow library.***



Step 8: Run "CTClassifier.py" to get a Graphical interface that allows you to pick
        a CT scan from the dataset and predict it's diagnosis in real time.

Step 9: Run "EnsembleTester.py" to test the ensemble classifier on the test set which consists of 374 images. Note that this
        classifier makes use of pre-trained and saved transfer learning models (Vgg16 and EfficientNetB5). You will be
        asked to choose between a hard and soft voting ensemble classifier. Type in "hard" for hard voting and "soft" for soft
        voting. It may take some time to run. It will display the confusion matrix for the classifier you chose and the evaluation
        metrics for the entire test set, such as accuracy, recall, etc.

Step 10: Run "EnsembleTrainer.py" to re-train the Vgg16 and EfficientNetB5 classifiers. It is recommended to only do this if
        you have a GPU on your computer as this is a very computationally heavy process. After training is complete, you will be
        asked if you want to overwrite the current saved trained models for Vgg16 and EfficientNetB5. Type "Y" for Yes and "N" for No.
        Be very careful when answering yes because you will lose the currently optimally trained models for Vgg16 and EfficientNetB5
        and overwrite them with your newly trained models.
