from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model
from tensorflow.keras import models
from tensorflow.keras import layers
from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.applications import vgg16
import os


class CTEnsembleTrain:

    def __init__(self):
        # initialise image datagenerators for training and validating
        self.train_datagenerator = ImageDataGenerator()
        self.validation_datagenerator = ImageDataGenerator()
        # fill train and test generators with training and validation set images, respectively
        self.fill_generators()

        # initialise CNNs to be used as base learners for ensemble
        models = []
        models.append(vgg16.VGG16(include_top=False,weights="imagenet"))
        models.append(efficientnet.EfficientNetB5(include_top=False, weights="imagenet"))
        self.models = models
        # prepare the models for transfer learning
        self.transfer_learning()

        # set names of base learner CNNs
        self.model_names = ["Vgg16","EfficientNetB5"]

    def fill_generators(self):
        # read training set
        self.train_datagenerator = ImageDataGenerator(preprocessing_function=None).flow_from_directory("SplitDataset/train",
                                                                     batch_size=16, class_mode="categorical",
                                                                     shuffle=True
                                                                     , target_size=(224, 224), seed=42)
        # read validation set
        validation_gen = ImageDataGenerator(preprocessing_function=None)
        self.validation_datagenerator = validation_gen.flow_from_directory("SplitDataset/val",
                                                                           batch_size=16, class_mode="categorical",
                                                                           shuffle=True
                                                                           , target_size=(224, 224), seed=42)



    def transfer_learning(self):
        for CNN_num in range(len(self.models)):

            # freeze the convolutional layers of the base predictors
            self.models[CNN_num].trainable = False

            model = models.Sequential()
            # add the convolutional layers to a new neural network
            model.add(self.models[CNN_num])

            # add global average pooling layer
            model.add(layers.GlobalAveragePooling2D(name="gap"))

            # add fully connected hidden layer
            model.add(layers.Dense(256, activation="relu"))
            # place a drop out between fully connected layer and output layer
            model.add(layers.Dropout(rate=0.3, name="dropout_out"))

            # add softmax output layer
            model.add(layers.Dense(2, activation="softmax", name="output"))

            # get model summary
            model.summary()

            # update the base predictors with new base predictors ready for transfer learning
            self.models[CNN_num] = model




    def train_models(self):

        for modelNum in range(len(self.models)):
            # display name of base learner that is training
            print("Training  " + self.model_names[modelNum])
            print()

            # compile model
            self.models[modelNum].compile(loss="categorical_crossentropy",
                                          optimizer=tf.optimizers.Adam(learning_rate=0.001)
                                          , metrics=["accuracy"])

            # create a callback that will adjust learning rate while training
            reducelr_callback = [ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                   patience=2, min_lr=0.00001,
                                                   verbose=1, mode='min')]

            # create a callback that can stop training early. Restores best weights learnt if training is stopped early
            earlystopping_callback = [EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)]

            # train the base predictor
            self.models[modelNum].fit(self.train_datagenerator, steps_per_epoch=None, epochs=1,
                                      validation_data=self.validation_datagenerator,
                                      validation_steps=None, verbose=1, shuffle=True, use_multiprocessing=True,
                                      workers=4
                                      , callbacks=[reducelr_callback, earlystopping_callback])

            print("--------------------------------------------------------------------------------------------------")
            print("--------------------------------------------------------------------------------------------------")

        # once all models are trained, ask if user wants to overwrite currently saved trained models
        overwrite_model = ""
        while True:
            overwrite_model = input("Do you want to overwrite the current saved models with the newly trained models? Type Y for Yes and N for No\>")

            if (overwrite_model.lower() == "y"):
                for num in range(len(self.models)):
                    print("overwriting the saved model for " + self.model_names[num] + "...")
                    save_model(model=self.models[num], filepath=self.model_names[num])
                    save_model(model=self.models[num], filepath=self.model_names[num])  # save trained models
                print("overwrite complete")

            if overwrite_model.lower() == "y" or overwrite_model.lower() == "n":
                break
            else:
                print("Please enter only Y for Yes and N for No")


if __name__== "__main__":
    if not(os.path.exists("SplitDataset")):
        print("SplitDataset Folder not found. Training cannot commence without the split data set. Please follow step 6 in the 'HowTo.txt' file.")
    else:
        trainer = CTEnsembleTrain()
        trainer.train_models()