from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os


class CTEnsembleTest:

    def __init__(self):
        # initialise test data generator
        self.test_datagenerator = ImageDataGenerator()
        # fill test data generator with test set images
        self.fill_generator()

        # initialise base classifier names
        self.model_names = ["Vgg16", "EfficientNetB5"]

        # extract the saved, trained models
        self.models = self.get_models()


    def get_models(self):
        loaded_models = []
        print("Loading Models...")
        for model_num in range(len(self.model_names)):
            # load the saved, trained models
            loaded_model = load_model(self.model_names[model_num])
            loaded_models.append(loaded_model)

        print("Finished Loading Models...")
        return loaded_models

    def fill_generator(self):
        self.test_datagenerator = ImageDataGenerator(preprocessing_function=None).flow_from_directory(
            "SplitDataset/test", batch_size=16, class_mode="categorical", shuffle=False, target_size=(224, 224))

        #print(self.test_datagenerator.labels)

    # test each trained model on the test set individually and get the accuracy
    def test_models(self):
        for trained_model in self.models:
            trained_model.evaluate(self.test_datagenerator)


    def get_hard_vote(self):
        print("Aquiring Hard Votes...")
        test_covid_pos_perc = [0] * len(self.test_datagenerator.labels) # list to store votes for COVID-19 Positive class

        test_covid_neg_perc = [0] * len(self.test_datagenerator.labels) # list to store votes for COVID-19 Negative class

        # for each trained model, get the predictions of each image in the test set
        # each prediction is a list with 2 probabilities - 1 for each possible prediction
        for model in self.models:
            # get vote prediction percents for each test image
            vote_percents = model.predict(self.test_datagenerator, verbose=0
                                          , batch_size=16)

            # get the number of votes for each of the 2 possible classes
            for i in range(0, len(vote_percents)):
                # compare probabilities for each class prediction
                if (vote_percents[i][0] >= vote_percents[i][1]):
                    test_covid_pos_perc[i] = (test_covid_pos_perc[i] + 1)
                else:
                    test_covid_neg_perc[i] = (test_covid_neg_perc[i] + 1)
        print()
        print("Determining Hard Voting predictions...")
        hard_vote_pred = []  # list to store hard voting prediction labels
        for j in range(0, len(test_covid_pos_perc)):

            # compare votes for each class. In case of a tie, predict COVID-19 Positive
            if test_covid_pos_perc[j] >= test_covid_neg_perc[j]:
                hard_vote_pred.append(0) # 0 is label for COVID-19 Positive Prediction
            else:
                hard_vote_pred.append(1) # 1 is label for COVID-19 Negative Prediction

        labels = self.test_datagenerator.labels # actual labels of test set


        # display confusion matrix for soft ensemble classifier
        self.printCM(labels, hard_vote_pred)
        print()
        print("Hard Voting Metrics")
        # display evaluation metrics for model performance
        eval_ensemble = self.evaluateEnsemble(labels, hard_vote_pred)
        eval_ensemble.get_metrics_report()



    def get_soft_vote(self):
        print("Aquiring Soft Votes...")
        test_covid_pos_perc = [0] * len(self.test_datagenerator.labels) # list to store votes for COVID-19 Positive class
        test_covid_neg_perc = [0] * len(self.test_datagenerator.labels) # list to store votes for COVID-19 Negative class

        for model in self.models:
            # get vote prediction percents for each test image
            vote_percents = model.predict(self.test_datagenerator, verbose=0 ,batch_size=16)

            # get the total sum of probability, from all models, for for each class, for each test image
            # each prediction is a list with 2 probabilities - 1 for each possible prediction
            for i in range(0, len(vote_percents)):
                test_covid_pos_perc[i] = (test_covid_pos_perc[i] + vote_percents[i][0])
                test_covid_neg_perc[i] = (test_covid_neg_perc[i] + vote_percents[i][1])
        print()
        print("Determining Soft Voting predictions...")
        soft_vote_pred = [] # store soft voting label predictions
        for j in range(0, len(test_covid_pos_perc)):

            # get average probability for each class, for each test image
            test_covid_pos_perc[j] = test_covid_pos_perc[j] / len(self.models)
            test_covid_neg_perc[j] = test_covid_neg_perc[j] / len(self.models)



            # compare average probability of each class. In case of a tie (very unlikely), predict COVID-19 Positive
            if test_covid_pos_perc[j] >= test_covid_neg_perc[j]:
                soft_vote_pred.append(0)  # 0 is positive label
            else:
                soft_vote_pred.append(1)  # 1 is negative label

        labels = self.test_datagenerator.labels # actual class labels

        # display confusion matrix for soft ensemble classifier
        self.printCM(labels, soft_vote_pred)

        print()
        print("Soft Voting Metrics")
        # display evaluation metrics for model performance
        eval_ensemble = self.evaluateEnsemble(labels,soft_vote_pred)
        eval_ensemble.get_metrics_report()


    # get confusion matrix using heatmap library
    def printCM(self ,true ,pred):
        cm = confusion_matrix(true, pred)
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, fmt='g',
                    ax=ax ,cmap="Greens")  # annot=True to annotate cells, ftm='g' to disable scientific notation

        # labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('')
        ax.xaxis.set_ticklabels(['COVID-19 Positive', 'COVID-19 Negative'])
        ax.yaxis.set_ticklabels(['COVID-19 Positive', 'COVID-19 Negative'])
        plt.show()



    # inner class for determining evaluation metrics
    class evaluateEnsemble:

        def __init__(self, truth, pred):
            self.TP, self.TN, self.FP, self.FN = self.get_metrics(truth, pred)

        def get_metrics(self, truth, pred):

            TP = 0
            TN = 0
            FP = 0
            FN = 0
            # 0 is positive , 1 is negative
            for i in range(len(truth)):
                if truth[i] == 0 and pred[i] == 0:
                    TP = TP + 1

                if truth[i] == 1 and pred[i] == 1:
                    TN = TN + 1

                if truth[i] == 0 and pred[i] == 1:
                    FN = FN + 1

                if truth[i] == 1 and pred[i] == 0:
                    FP = FP + 1

            return (TP, TN, FP, FN)

        def get_recall(self):
            return self.TP / (self.TP + self.FN)

        def get_precision(self):
            return self.TP / (self.TP + self.FP)

        def get_specificity(self):
            return self.TN / (self.TN + self.FP)

        def get_F1Score(self):
            return (2 * self.get_precision() * self.get_recall()) / (self.get_precision() + self.get_recall())

        def get_acc(self):
            return (self.TP + self.TN) / (self.TP + self.TN + self.FN + self.FP)

        def get_metrics_report(self):
            print("TP: " + str(self.TP))
            print("TN: " + str(self.TN))
            print("FP: " + str(self.FP))
            print("FN: " + str(self.FN))
            print("Accuracy: " + str(self.get_acc()))
            print("Recall: " + str(self.get_recall()))
            print("Precision: " + str(self.get_precision()))
            print("Specificity: " + str(self.get_specificity()))
            print("F1 Score: " + str(self.get_F1Score()))


if __name__== "__main__":
    if not(os.path.exists("SplitDataset")):
        print("SplitDataset Folder not found. Testing cannot commence without the split data set. Please follow step 6 in the 'HowTo.txt' file.")
    elif not(os.path.exists("Vgg16")) or not(os.path.exists("EfficientNetB5")):
        print("One or more trianed models for Vgg16 and EfficientNetB5 are not found. Please follow step 2 of the 'HowTo.txt' file.")
    else:
        tester = CTEnsembleTest()

        while True:
            ensemble_option = input("Enter the ensemble method that you want to test: Enter hard (for hard voting) or soft (for soft voting) only\n>")

            if (ensemble_option.lower() == "hard"):
                tester.get_hard_vote()
                break

            elif (ensemble_option.lower() == "soft"):
                tester.get_soft_vote()
                break

            else:
                print("Please enter only hard (for hard voting) or soft (for soft voting)")
