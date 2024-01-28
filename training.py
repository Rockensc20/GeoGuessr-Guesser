from keras.layers import Dense, Flatten, Dropout, RandomFlip, RandomRotation, RandomZoom, GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img,ImageDataGenerator, img_to_array
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import pandas as pd
import datetime
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, RandomFlip, RandomRotation, RandomZoom, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import numpy as np

import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt

# Image loading parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
TRAIN_DIR ="scaled_images_splitted/train"
VAL_DIR ="scaled_images_splitted/val"
BATCH_SIZE = 32

# Model hyperparameters
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'

# Training parameters
LEARN_RATE = 0.01
EPOCHS = 30

# Check whether GPU is available to be used
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

##############################
#### Data loading section ####
##############################
loaded_train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels = "inferred",
    label_mode = "categorical",
    color_mode = "rgb",
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle = True,
    seed=42
)

loaded_val_data = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels = "inferred",
    label_mode = "categorical",
    color_mode = "rgb",
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle = True,
    seed=42
)

#preprocessing_layer = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)
processed_train_data = loaded_train_data.map(lambda x, y: (preprocess_input(x), y))
processed_val_data = loaded_val_data.map(lambda x, y: (preprocess_input(x), y))

# Use TensorFlow's AUTOTUNE to automatically adjust the number of parallel calls during data preprocessing
AUTOTUNE = tf.data.AUTOTUNE

# Cache, shuffle, and prefetch the training dataset for optimized performance
train_data = loaded_train_data.cache().prefetch(buffer_size=AUTOTUNE)
val_data = loaded_val_data.cache().prefetch(buffer_size=AUTOTUNE)

##############################
### Model buidling section ###
##############################

# Load pre-trained ResNet50v2 model with the ImageNet weights, fully connected top layer isn't included because we will do transfer learning
densenet_model = DenseNet121(
    include_top=False,
    weights="imagenet",
    classifier_activation="softmax",
    input_shape=(IMG_HEIGHT, IMG_WIDTH,3)
)

# Freeze the layers in the pretrained model as we do not want to train them anymore
for layer in densenet_model.layers[:149]:
  layer.trainable = False
for layer in densenet_model.layers[149:]:
  layer.trainable = True

# Create our transfer model that will use the pretrained resnet as input
transfer_model = Sequential()
transfer_model.add(densenet_model)
transfer_model.add(GlobalAveragePooling2D())
transfer_model.add(Dense(512, activation='relu'))
transfer_model.add(Dropout(0.7))
transfer_model.add(BatchNormalization())
transfer_model.add(Dense(256, activation='relu'))
transfer_model.add(Dropout(0.5))
transfer_model.add(BatchNormalization())
transfer_model.add(Dense(128, activation='relu'))
transfer_model.add(Dropout(0.5))
transfer_model.add(BatchNormalization())
transfer_model.add(Dense(64, activation='relu'))
transfer_model.add(Dropout(0.3))
transfer_model.add(BatchNormalization())
transfer_model.add(Dense(2,activation='softmax'))
transfer_model.summary()

# Instantiate our desired optimzer, could be grid searched do find a more optimal configuration
adam = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Compile model to use for training
transfer_model.compile(
    optimizer = adam,
    loss =  'categorical_crossentropy' ,
    metrics=["accuracy"]
)


##############################
### Model training section ###
##############################

# Tools for plots, early stopping & checkpointing the best model in ../working dir & restoring that as our model for prediction
#train_plot_loss = PlotLossesCallback()

train_early_stopper = EarlyStopping(
    monitor = 'val_loss',
    patience = 10,
    )
train_checkpointer = ModelCheckpoint(
    filepath = 'models/dense_categorical_model_best_adam.hdf5',
    monitor = 'val_loss',
    save_best_only = True,
    mode = 'auto',
    verbose=1
)

resnet50v2_transfer_fit_history = transfer_model.fit(
        train_data,
        epochs = 2,
        validation_data=val_data,
        callbacks=[train_early_stopper,train_checkpointer]
)