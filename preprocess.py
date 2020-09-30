import os
import sys

# linear algebra
import numpy as np

# data processing 
import pandas as pd
from tensorflow.keras.utils import to_categorical, plot_model
# data visualizations
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from cv2 import imread
from cv2 import resize
import cv2
# File format/ file system packages
import json
import h5py
from scipy.io import loadmat
from unpack import *

def crop_and_resize(image, img_size):
    """ Crop and resize an image
    """
    image_data = imread(image['filename'])
    crop = image_data[image['y0']:image['y1'], image['x0']:image['x1'], :]
    return resize(crop, img_size)


def create_dataset(df, img_size):
    """ Helper function for converting images into a numpy array
    """
    X = np.zeros(shape=(df.shape[0], img_size[0], img_size[0], 3), dtype='uint8')
    y = np.full((df.shape[0], 5), 10, dtype=int)
    
    # Iterate over all images in the pandas dataframe (slow!)
    for i, (index, image) in enumerate(df.iterrows()):
        
        # Get the image data
        X[i] = crop_and_resize(image, img_size)
        
        # Get the label list as an array
        labels = np.array((image['labels']))
        #print (labels.shape)
        # Store 0's as 0 (not 10)
        labels[labels==10] = 0
        
        # Embed labels into label array
        y[i,0:labels.shape[0]] = labels
        
    # Return data and labels   
    return X, y


def dict_to_df(image_bounding_boxes, path):
    """ Helper function for flattening the bounding box dictionary
    """
    # Store each bounding box
    boxes = []

    # For each set of bounding boxes
    for image in image_bounding_boxes:

        # For every bounding box
        for bbox in image['boxes']:

            # Store a dict with the file and bounding box info
            boxes.append({
                    'filename': path + image['filename'],
                    'label': bbox['label'],
                    'width': bbox['width'],
                    'height': bbox['height'],
                    'top': bbox['top'],
                    'left': bbox['left']})

    return pd.DataFrame(boxes)

def get_img_size(filepath):
    """Returns the image size in pixels given as a 2-tuple (width, height)
    """
    image = imread(filepath)
    return (image.shape)

def get_img_sizes(folder):
    """Returns a DataFrame with the file name and size of all images contained in a folder
    """
    image_sizes = []
    
    images = [img for img in os.listdir(folder) if img.endswith('.png')]
    
    # Get image size of every individual image
    for image in images:
        h,w,_ = get_img_size(folder + image)
        image_size = {'filename': folder + image, 'image_width': w, 'image_height': h}
        image_sizes.append(image_size)
        
    # Return results as a pandas DataFrame
    return pd.DataFrame(image_sizes)

def rgb2gray(images):
    return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)

def random_sample(N, K):
    """Return a boolean mask of size N with K selections
    """
    mask = np.array([True]*K + [False]*(N-K))
    np.random.shuffle(mask)
    return mask

def process_data():

    
    bbox_file = './Data/bounding_boxes.csv'
    

    if not os.path.isfile(bbox_file):
        train_bbox = parse_info('./Data/Format1/train/digitStruct.mat')
        test_bbox = parse_info('./Data/Format1/test/digitStruct.mat')
        extra_bbox = parse_info('./Data/Format1/extra/digitStruct.mat')

    
        train_df = dict_to_df(train_bbox, './Data/Format1/train/')
        test_df = dict_to_df(test_bbox, './Data/Format1/test/')
        extra_df = dict_to_df(extra_bbox, './Data/Format1/extra/')

        print("Training", train_df.shape)
        print("Test", test_df.shape)
        print('')

        # Concatenate all the information in a single file
        df = pd.concat([train_df, test_df, extra_df])

        print("Combined", df.shape)

        # Write dataframe to csv
        df.to_csv(bbox_file, index=False)

        del train_df, test_df, train_bbox, test_bbox, extra_df, extra_bbox

    else:
        # Load preprocessed bounding boxes
        df = pd.read_csv(bbox_file)

    df.rename(columns={'left': 'x0', 'top': 'y0'}, inplace=True)
    df['x1'] = df['x0'] + df['width']
    df['y1'] = df['y0'] + df['height']
    # Apply the aggegration
    df = df.groupby('filename').agg(
            x0 = pd.NamedAgg(column='x0', aggfunc=min),
            y0 = pd.NamedAgg(column='y0', aggfunc=min),
            x1 = pd.NamedAgg(column='x1', aggfunc=max),
            y1 = pd.NamedAgg(column='y1', aggfunc=max),
            labels = pd.NamedAgg(column='label', aggfunc=lambda x: list(x)),
            num_digits = pd.NamedAgg(column='label', aggfunc="count")
            ).reset_index()
    df['x_inc'] = ((df['x1'] - df['x0']) * 0.30) / 2.
    df['y_inc'] = ((df['y1'] - df['y0']) * 0.30) / 2.

    df['x0'] = (df['x0'] - df['x_inc']).astype('int')
    df['y0'] = (df['y0'] - df['y_inc']).astype('int')
    df['x1'] = (df['x1'] + df['x_inc']).astype('int')
    df['y1'] = (df['y1'] + df['y_inc']).astype('int')

    train_sizes = get_img_sizes('./Data/Format1/train/')
    test_sizes = get_img_sizes('./Data/Format1/test/')
    extra_sizes = get_img_sizes('./Data/Format1/extra/')

    # Concatenate all the information in a single file
    image_sizes = pd.concat([train_sizes, test_sizes, extra_sizes])

    # Delete old dataframes
    del train_sizes, test_sizes, extra_sizes
    print("Bounding boxes", df.shape)
    print("Image sizes", image_sizes.shape)
    print('')

    # Inner join the datasets on filename
    df = pd.merge(df, image_sizes, on='filename', how='inner')

    print("Combined", df.shape)

    # Delete the image size df
    del image_sizes

    # Store checkpoint
    df.to_csv("./Data/image_data1.csv", index=False)
    
    #df = pd.read_csv("./Data/image_data.csv")

    df.loc[df['x0'] < 0, 'x0'] = 0
    df.loc[df['y0'] < 0, 'y0'] = 0
    df.loc[df['x1'] > df['image_width'], 'x1'] = df['image_width']
    df.loc[df['y1'] > df['image_height'], 'y1'] = df['image_height']

    df = df[df.num_digits < 6]

    image_size = (32, 32)

    # Get cropped images and labels (this might take a while...)
    X_train, y_train = create_dataset(df[df.filename.str.contains('train')], image_size)
    X_test, y_test = create_dataset(df[df.filename.str.contains('test')], image_size)
    X_extra, y_extra = create_dataset(df[df.filename.str.contains('extra')], image_size)

    # We no longer need the dataframe
    del df
    print("Training", X_train.shape, y_train.shape)
    print("Test", X_test.shape, y_test.shape)
    print('Extra', X_extra.shape, y_extra.shape)

    # Pick 8000 training and 2000 extra samples
    sample1 = random_sample(X_train.shape[0], 5000)
    sample2 = random_sample(X_extra.shape[0], 2000)

    # Create valdidation from the sampled data
    X_val = np.concatenate([X_train[sample1], X_extra[sample2]])
    y_val = np.concatenate([y_train[sample1], y_extra[sample2]])

    # Keep the data not contained by sample
    X_train = np.concatenate([X_train[~sample1], X_extra[~sample2]])
    y_train = np.concatenate([y_train[~sample1], y_extra[~sample2]])

    # Moved to validation and training set
    #del X_extra, y_extra 

    print("Training", X_train.shape, y_train.shape)
    print('Validation', X_val.shape, y_val.shape)
    print('Extra', X_extra.shape, y_extra.shape)

    h5f = h5py.File('./Data/SVHN_multi-2_digit_rgb.h5', 'w')

    # Store the datasets
    h5f.create_dataset('X_train', data=X_train)
    h5f.create_dataset('y_train', data=y_train)
    h5f.create_dataset('X_test', data=X_test)
    h5f.create_dataset('y_test', data=y_test)
    h5f.create_dataset('X_val', data=X_val)
    h5f.create_dataset('y_val', data=y_val)

    # Close the file
    h5f.close()

    X_train = rgb2gray(X_train).astype(np.float32)
    X_test = rgb2gray(X_test).astype(np.float32)
    X_val = rgb2gray(X_val).astype(np.float32)

    # Calculate the mean on the training data
    train_mean = np.mean(X_train, axis=0)

    # Calculate the std on the training data
    train_std = np.std(X_train, axis=0)

    # Subtract it equally from all splits
    train_norm = (X_train - train_mean) / train_std
    test_norm = (X_test - train_mean)  / train_std
    val_norm = (X_val - train_mean) / train_std

    # Create file
    h5f = h5py.File('./Data/SVHN_multi-2_digit_norm_grayscale.h5', 'w')

    # Store the datasets
    h5f.create_dataset('X_train', data=train_norm)
    h5f.create_dataset('y_train', data=y_train)
    h5f.create_dataset('X_test', data=test_norm)
    h5f.create_dataset('y_test', data=y_test)
    h5f.create_dataset('X_val', data=val_norm)
    h5f.create_dataset('y_val', data=y_val)

    # Close the file
    h5f.close()

process_data()

