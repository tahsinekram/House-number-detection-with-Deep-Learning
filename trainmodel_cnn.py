from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
from tensorflow.keras.models import model_from_json, Model
import json
from model import cnn_model, base_model, VGG16Model
import numpy as np
import cv2
import h5py

def train_model(model_instance:base_model, model_json_file, weights_file, grayscale=True):
    h5f = h5py.File('./Data/SVHN_multi_digit_norm_grayscale.h5', 'r')

    # Load the training, test and validation set
    X_train = h5f['X_train'][:]
    y_train = h5f['y_train'][:]
    X_test = h5f['X_test'][:]
    y_test = h5f['y_test'][:]
    X_val = h5f['X_val'][:]
    y_val = h5f['y_val'][:]
    h5f.close()
    model = model_instance
    model.fit_model(X_train, y_train, X_val, y_val)
    model.save_model(model_json_file, weights_file)


train_model(cnn_model(), "cnn_model.json", "cnn_w.h5")
