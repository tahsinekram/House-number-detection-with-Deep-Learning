from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer,LeakyReLU, ReLU, Conv2D, MaxPool2D, Dense, Flatten, BatchNormalization, Activation, Dropout, Input, LeakyReLU, ZeroPadding2D, UpSampling2D
from tensorflow.keras import callbacks
from time import time
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from hyperas import optim
from hyperas.distributions import choice, uniform
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import json
import cv2
import h5py
from hyperopt import Trials, STATUS_OK, tpe, rand

def data(dfile):

    h5f = h5py.File(dfile, 'r')

    # Load the training, test and validation set
    X_train = h5f['X_train'][:]
    y_train = h5f['y_train'][:]
    X_test = h5f['X_test'][:]
    y_test = h5f['y_test'][:]
    X_val = h5f['X_val'][:]
    y_val = h5f['y_val'][:]
    h5f.close()

    return X_train, y_train, X_val, y_val, X_test, y_test


class base_model:

    history=None
    model=None
    acc_plt = None
    loss_plt = None

    def save_model(self, model_json_file, weights_file):
        """
        Saves the model and the weights
        :param model_json_file: path
        :param weights_file: path
        :return: None
        """
        self.model.save_weights(weights_file)
        model_json = self.model.to_json()
        with open(model_json_file, "w+") as file:
            file.write(model_json)

    @staticmethod
    def predict(model_json, weights, X):
        """
        Predicts the output given an input
        :param model_json: custom_model as a json
        :param weights: h5 containing weights
        :param X: input to be predicted
        :return: predicted output
        """
        with open(model_json, 'r') as json_file:
            model_json = json.load(json_file)
            loaded_model = model_from_json(json.dumps(model_json))
        
        loaded_model.load_weights(weights)
        Y = loaded_model.predict(X)
        return Y

class cnn_model(base_model):

    def __init__(self):
        

        input_layer = Input(shape=(32,32,1), name='input_layer')
        x=Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 1))(input_layer)
        x=BatchNormalization()(x)
        x=Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x=MaxPool2D(pool_size = (2, 2))(x)
        x=Dropout(0.4)(x)
        
        x=Conv2D(64, kernel_size=(3,3), activation='relu')(x)
        x=BatchNormalization()(x)
        x=Conv2D(64, kernel_size=(3,3), activation='relu')(x)
        x=MaxPool2D(pool_size = (2, 2))(x)
        x=Dropout(0.4)(x)

        x=Conv2D(128, kernel_size=(3,3), activation='relu')(x)
        x=BatchNormalization()(x)
        x=Conv2D(128, kernel_size=(3,3), activation='relu')(x)
        x=MaxPool2D(pool_size = (2, 2))(x)
        x=Dropout(0.4)(x)
        
        x=Flatten()(x)

        x=Dense(128, activation='relu')(x)
        
        d1=Dense(11,activation='softmax')(x)
        d2=Dense(11,activation='softmax')(x)
        d3=Dense(11,activation='softmax')(x)
        d4=Dense(11,activation='softmax')(x)
        d5=Dense(11,activation='softmax')(x)

        lst = [d1,d2,d3,d4,d5]

        self.model = Model(input_layer,lst)


    def fit_model(self, X_train, y_train, X_val, y_val):

        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)

        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5)
        model_checkpoint = callbacks.ModelCheckpoint('multi-digit_cnn_new.h5',save_best_only=True)

        optimizer = Adam(lr=1e-3,amsgrad=True)
        tb = callbacks.TensorBoard(log_dir="ccnlogs/{}".format(time()))
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        self.history = self.model.fit(X_train, [y_train[:,0],y_train[:,1],y_train[:,2],y_train[:,3],y_train[:,4]], batch_size=512,
                                           epochs=12, shuffle = True, 
                                           validation_data = (X_val,[y_val[:,0],y_val[:,1],y_val[:,2],y_val[:,3],y_val[:,4]]),
                                           callbacks=[early_stopping, model_checkpoint])


    def evaluate(self,X_test, Y_test, mod=False, model_json_file="file.json", weights="weights"):
        if mod:
            with open(model_json_file, 'r') as json_file:
                model_json = json.load(json_file)
                loaded_model = model_from_json(json.dumps(model_json))

            loaded_model.load_weights(weights)
            optimizer = Adam(lr=1e-3,amsgrad=True)
            loaded_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            print(loaded_model.evaluate(X_test,[Y_test[:,0],Y_test[:,1],Y_test[:,2],Y_test[:,3],Y_test[:,4]], verbose =1))
            
        else:
            print (self.model.evaluate(X_test,[Y_test[:,0],Y_test[:,1],Y_test[:,2],Y_test[:,3],Y_test[:,4]], verbose =1))


if __name__ == '__main__':


    mod = cnn_model()
  
    __, __, __, __, X_test, Y_test = data('./Data/SVHN_multi_digit_norm_grayscale.h5')
    mod.evaluate(X_test, Y_test, True, "./Data/cnn_model-ng1.json","./Data/cnn_w-ng1.h5")

