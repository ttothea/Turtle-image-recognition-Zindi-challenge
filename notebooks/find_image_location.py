import pathlib
import IPython.display as display
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import csv
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#Load csv-data
image_dir = "../images/"
train_data = pd.read_csv('../data/train_corrected.csv')
train_data.image_id = train_data.image_id.apply(lambda x: x.strip()+".JPG")
test_data = pd.read_csv('../data/test_corrected.csv')
test_data.image_id = test_data.image_id.apply(lambda x: x.strip()+".JPG")

#Get unique_turtle_ids from train.csv
unique_turtle_ids = list(train_data['turtle_id'].unique())
#Add category for new turtle for test set
unique_turtle_ids.append("new_turtle")
#Get number of images for train/test split
split = 0.7
lines = round(len(train_data)*split)
length_data = len(train_data)

#We set some parameters for the model
HEIGHT = 224 #image height
WIDTH = 224 #image width
CHANNELS = 3 #image RGB channels
CLASS_NAMES = ['left', 'top', 'right']
NCLASSES = 3
BATCH_SIZE = 32
SHUFFLE_BUFFER = 10 * BATCH_SIZE
AUTOTUNE = tf.data.experimental.AUTOTUNE

TRAINING_SIZE = lines
VALIDATION_SIZE = length_data - lines                    
VALIDATION_STEPS = VALIDATION_SIZE // BATCH_SIZE

# flow_from_directory : Takes the path to a directory & generates batches of augmented data.
# use "rescale" to scale array of original image pixel values to be between [0,1] and specify the parameter rescale=1./255.

def preprocess(augment = True):
    if augment == True:
        train_datagen = ImageDataGenerator(
                rotation_range     = 40,
                width_shift_range  = 0.2,
                height_shift_range = 0.2,
                rescale            = 1./255,
                shear_range        = 0.2,
                zoom_range         = 0.2,
                horizontal_flip    = False,
                fill_mode          = 'nearest')

        test_datagen = ImageDataGenerator(rescale=1./255)
    
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen  = ImageDataGenerator(rescale=1./255)
        
    return train_datagen, test_datagen

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
#x_col value : which will be the name of column(in dataframe) having file names
#y_col value : which will be the name of column(in dataframe) having class/label

def use_image_generator(train_datagen, test_datagen, training=True): 
    
    if training == True:
        # Augment and scale images for training
        train_generator = train_datagen.flow_from_dataframe(dataframe =train_data[0:lines], 
                directory   = image_dir,
                x_col       = "image_id" ,
                y_col       = "image_location",
                target_size = (HEIGHT, WIDTH),
                batch_size  = BATCH_SIZE,
                classes     = CLASS_NAMES,
                class_mode  = 'categorical',
                shuffle     = True)
                #save_to_dir="output/",  if you wanna save the cropped images
                #save_prefix="",
                #save_format='png')

        # Scale images for validation
        validation_generator = test_datagen.flow_from_dataframe(dataframe = train_data[lines:], 
                directory    = image_dir,
                x_col        = "image_id",
                y_col        = "image_location",
                target_size  = (HEIGHT, WIDTH),
                batch_size   = BATCH_SIZE,
                classes      = CLASS_NAMES,
                class_mode   = 'categorical',
                shuffle      = True)
        
        return train_generator, validation_generator
    
    else:
        # Scale images for testing, no target provided and returned
        test_generator = test_datagen.flow_from_dataframe(dataframe = test_data, 
                directory   = image_dir,
                x_col       = "image_id",
                target_size = (HEIGHT, WIDTH),
                batch_size  = BATCH_SIZE,
                class_mode  = None,
                shuffle     = False)
            
        return test_generator
