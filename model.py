#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 19:53:19 2017

@author: KrazyK
"""
#import os
import csv
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn

## Create flag variables for use on the command line
#flags = tf.app.flags
#FLAGS = flags.FLAGS
#
## Command line flags
#flags.DEFINE_string('load', './model.h5', "Load model string (.h5)")
#flags.DEFINE_string('save', './model_new.h5', "Save model string (.h5)")
#flags.DEFINE_integer('epochs', 10, 'Number of epochs to train')
#flags.DEFINE_integer('batch_size', 6, 'Batch size for training')

# Import csv data file containing the path to all training images
samples = []
removed = 0
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if float(line[3])!=0: # Only imports data with steering angle not equal to zero
            samples.append(line)
        else:
            removed+=1 # Count removed zero steer angle data
    print('Number of lines in csv = ', len(samples))
    print('Number of samples removed with zero steering value = ', removed)

# Split the data into 80% training and 20% validation sets
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Generator for training. Creates data in batches as needed by train().
from sklearn.utils import shuffle
def generator(samples, batch_size):
    num_samples = len(samples)
    once=0
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            # Ingest left, center, and right camera images and add steering corrections
            for batch_sample in batch_samples:
                for i in range(3):
                    name = './data/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    if i==0:
                        angle = float(batch_sample[3])#; print('center', measurement)
                    elif i==1:
                        angle = float(batch_sample[3]) + 0.05#; print('left', measurement)
                    elif i==2:
                        angle = float(batch_sample[3]) - 0.05#; print('right', measurement)

                    images.append(image)
                    angles.append(angle)
                    
                    # Store flipped images and steer angels into variables
                    augmented_images, augmented_angles = [], []
                    for image, angle in zip(images, angles):
                        augmented_images.append(image)
                        augmented_angles.append(angle)
                        augmented_images.append(cv2.flip(image,1))
                        augmented_angles.append(angle * -1.0)
                        
            # Show example images
            #if once==0:
                import random
                plt.figure()
                plt.imshow(augmented_images[random.randint(0, len(augmented_images)/3)], interpolation='none', cmap='gray')
                plt.colorbar()
                plt.savefig('./examples/image'+str(once)+'.png', bbox_inches='tight')
                once+=1
            
            # Convert to numpy array
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)


             
batch_size = FLAGS.batch_size; print('Batch size = ', batch_size,' * 3cams',
                      ' * 2 for flips = ', 6*batch_size)
epochs = FLAGS.epochs; print('Epochs = ', epochs)


ch, row, col = 3, 160, 320  # Untrimmed image format


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D as Conv2D
#from keras.layers.pooling import MaxPooling2D


def create_model():
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))
    model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(row, col, ch)))
    #model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Conv2D(24,5,5, subsample=(2,2), activation='relu', init='normal'))
    model.add(Dropout(.25))
    model.add(Conv2D(36,5,5, subsample=(2,2), activation='relu', init='normal'))
    model.add(Dropout(.25))
    model.add(Conv2D(48,5,5, subsample=(2,2), activation='relu', init='normal'))
    model.add(Dropout(.25))
    #model.add(MaxPooling2D())
    model.add(Conv2D(64,3,3, activation='relu', init='normal'))
    model.add(Dropout(.25))
    model.add(Conv2D(64,3,3, activation='relu', init='normal'))
    model.add(Dropout(.25))
    #model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(100, init='normal'))
    model.add(Dropout(.25))
    model.add(Dense(50, init='normal'))
    model.add(Dropout(.25))
    model.add(Dense(10, init='normal'))
    model.add(Dropout(.25))
    model.add(Dense(1, init='normal'))
    
    return model
    
def train(model):
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)
    model.compile(optimizer='adam', loss='mae')
    
    print('\n Number of training samples  = {} \n'.format(len(train_samples)))
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                        validation_data=validation_generator,
                        nb_val_samples=len(validation_samples),
                        nb_epoch=epochs)
    if FLAGS.save:
        model.save(FLAGS.save)
        print('\n Model saved to..... {} \n'.format(FLAGS.save))
    else:
        model.save('model.h5')
        print('\n Model saved to..... model.h5 \n')
        
# Load model from file 
from keras.models import load_model    
def load_trained_model(weights_path):
   model = load_model(weights_path)
   return model

# Load model from a previously trained model or create new model
if FLAGS.load: 
    print('\n Model loaded from {} \n'.format(FLAGS.load))
    model = load_trained_model(FLAGS.load)
else:
    print('\n Creating new model.... \n')
    model = create_model()

# Train model with new data
train(model)



