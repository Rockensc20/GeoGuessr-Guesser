import tensorflow as tf
import pandas as pd
import datetime
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, RandomFlip, RandomRotation, RandomZoom, GlobalAveragePooling2D,BatchNormalization
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.resnet_v2 import preprocess_input
    
IMG_HEIGHT = 224
IMG_WIDTH = 224

model_weights = 'models/resnet.hdf5'
model_name = 'ResNet50'

def load_resnet_model():
    resnet50v2_model = ResNet50V2(
        include_top=False,
        weights='imagenet',
        pooling='avg',
        input_shape=(IMG_HEIGHT, IMG_WIDTH,3)
    )

    # Freeze the layers in the pretrained model as we do not want to train them anymore
    for layer in resnet50v2_model.layers:
        layer.trainable = False

    # Create our transfer model that will use the pretrained resnet as input
    transfer_model = Sequential()
    transfer_model.add(resnet50v2_model)
    transfer_model.add(Flatten())
    transfer_model.add(Dense(512, activation='relu'))
    transfer_model.add(BatchNormalization())
    transfer_model.add(Dense(256, activation='relu'))
    transfer_model.add(BatchNormalization())
    transfer_model.add(Dropout(0.5))
    transfer_model.add(Dense(128, activation='relu'))
    transfer_model.add(Dropout(0.3))
    transfer_model.add(BatchNormalization())
    transfer_model.add(Dense(64, activation='relu'))
    transfer_model.add(Dropout(0.3))
    transfer_model.add(BatchNormalization())
    transfer_model.add(Dense(2,activation='softmax'))
    transfer_model.summary()

    # Instantiate our desired optimzer, could be grid searched do find a more optimal configuration
    adam = tf.keras.optimizers.Adam(lr=0.0001)

    # Compile model to use for training
    transfer_model.compile(
        optimizer = tf.keras.optimizers.SGD(lr = 0.0001, momentum = 0.9, nesterov = True),
        loss = 'categorical_crossentropy',
        metrics=["accuracy"]
    )

    return transfer_model

def make_predictions(resnet_model, test_ds):
    true_labels = np.concatenate(list(test_ds.map(lambda x, y: y).as_numpy_iterator()))
    true_labels_int = true_labels

    # Get model predictions
    predictions = resnet_model.predict(test_ds)
    print(predictions)

    # Convert predictions to binary (0 or 1) based on a threshold
    predictions_binary = (predictions[:, 1] > 0.5).astype(np.int32)

    # Compute confusion matrix
    confusion_matrix = tf.math.confusion_matrix(labels=true_labels_int, predictions=predictions_binary, num_classes=2)

    # Normalize confusion matrix to percentage
    cm_percentage = confusion_matrix / tf.math.reduce_sum(confusion_matrix, axis=1, keepdims=True) * 100

    # Convert numerical labels to string labels for the heatmap
    label_mapping = {0: 'Asia', 1: 'Europe'}
    true_labels_str = [label_mapping[label] for label in true_labels_int]

    model_accuracy = np.mean(predictions_binary == true_labels_int)

    return cm_percentage.numpy(), model_accuracy

def print_heatmap(cm_percentage, model_accuracy):
    class_names = ['Asia', 'Europe']
    heatmap_df = pd.DataFrame(cm_percentage, index=class_names, columns=class_names)

    # Create the heatmap using Plotly Express imshow
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns,
            y=heatmap_df.index,
            coloraxis='coloraxis',
            zmax=100,
            zmin=0
        )
    )
    fig.update_xaxes(title_text='Predicted')
    fig.update_yaxes(title_text='True')

    fig.update_layout(coloraxis=dict(colorscale='Blues'), title_text='{} : {:.2f}%'.format(model_name, model_accuracy * 100))
    fig.show()

def load_data():
    # Define the directory containing the subfolders
    data_directory = "scaled_images_splitted"

    # Load the test dataset
    test_data = tf.keras.utils.image_dataset_from_directory(
        directory=data_directory + "/test",
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=32,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        shuffle=False,
        seed=42
    )

    # Scale data for all values to be between 0 and 1
    test_data = test_data.map(lambda x, y: (preprocess_input(x), y))

    # Cache and prefetch datasets
    test_data = test_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return test_data


if __name__ == '__main__':
    # load data
    test_ds = load_data()

    # load model
    resnet_model = load_resnet_model()
    resnet_model = tf.keras.models.load_model(model_weights, compile=False)
    resnet_model.compile(
        optimizer = tf.keras.optimizers.SGD(lr = 0.0001, momentum = 0.9, nesterov = True),
        loss = 'categorical_crossentropy',
        metrics=["accuracy"]
    )

    # make prediction and get confusion matrix
    cm_percentage, model_accuracy = make_predictions(resnet_model, test_ds)

    # print confusion matrix
    print_heatmap(cm_percentage, model_accuracy)