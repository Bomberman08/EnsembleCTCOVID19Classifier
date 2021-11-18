# EnsembleCTCOVID19Classifier
Honours Project
Note: you will need to have a python compiler installed on your computer to run this software. This software makes use
      of pre-trained Vgg16 and EfficientNetB5 transfer learning models to form hard and soft voting ensemble classifiers.
      The final classifier software makes use of a graphical interface and only uses the soft voting ensemble classifier to
      classify CT scans in real time.

Step 1: Download and unzip the source code files and trained models from Github using the link provided in the Github.txt file.

Step 2: Download the dataset using the link provided in the Dataset.txt file. 

Step 3: Move the downloaded dataset to the directory that you saved the source code files.

Step 4: Open a python IDE or a terminal to run the source code files.

Step 5: Run the source code file "Prelim.py" which will install the relavent libraries needed to run the code
        and will also randomly split the dataset into training, validation and test sets. Make sure to wait until
        the code has completed running and ensure you have internet connection. 

Step 6: You have 3 files you can run from here. See step 7 for running "CTClassifier.py", see step 8 for running "EnsembleTester.py",
         and see step 9 for running "EnsembleTrainer.py". 


***Note that you may see Warnings in your terminal when running each of the 3 python files.
This does not mean there are errors in the software ; these are simply warnings automatically displayed by the tensorflow library.***



Step 7: Run "CTClassifier.py" to get a Graphical interface that allows you to pick
        a CT scan from the dataset and predict it's diagnosis in real time.

Step 8: Run "EnsembleTester.py" to test the ensemble classifier on the test set which consists of 374 images. Note that this
        classifier makes use of pre-trained and saved transfer learning models (Vgg16 and EfficientNetB5). You will be
        asked to choose between a hard and soft voting ensemble classifier. Type in "hard" for hard voting and "soft" for soft
        voting. It may take some time to run. It will display the confusion matrix for the classifier you chose and the evaluation
        metrics for the entire test set, such as accuracy, recall, etc.

Step 9: Run "EnsembleTrainer.py" to re-train the Vgg16 and EfficientNetB5 classifiers. It is recommended to only do this if
        you have a GPU on your computer as this is a very computationally heavy process. After training is complete, you will be
        asked if you want to overwrite the current saved trained models for Vgg16 and EfficientNetB5. Type "Y" for Yes and "N" for No.
        Be very careful when answering yes because you will lose the currently optimally trained models for Vgg16 and EfficientNetB5
        and overwrite them with your newly trained models.
