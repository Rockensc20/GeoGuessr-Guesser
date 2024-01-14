import tensorflow as tf
import pandas as pd
import datetime
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, RandomFlip, RandomRotation, RandomZoom, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
    
img_height = 224
img_width = 224

def load_data():
    loaded_data = tf.keras.utils.image_dataset_from_directory(
        "scaled_images",
        labels = "inferred",
        label_mode = "int",
        color_mode = "rgb",
        batch_size=32,
        image_size=(img_height, img_width),
        shuffle = True,
        seed=42
    )

    # scale data for all values to be between 0 and 1 in order for better performance
    scaled_data = loaded_data.map(lambda x,y: (x/255, y))

    # partion data for training, validation and a heldback dataset
    dataset_length = len(scaled_data)
    train_size = int(dataset_length*0.7)
    val_size = int(dataset_length*0.2)
    test_size = int(dataset_length*0.1)

    train_ds = scaled_data.take(train_size)
    val_ds = scaled_data.skip(train_size).take(val_size)
    test_ds = scaled_data.skip(train_size+val_size).take(test_size)
    test_ds_labels = test_ds

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, test_ds, test_ds_labels

# load data
train_ds, test_ds, test_ds_labels = load_data()

y_values = []

for x, y in test_ds:
    arr = y.numpy().flatten()
    for i in arr:
        if i == 0:
            y_values.append('Asia')
        else:
            y_values.append('Europe')

iter = 0
model_name = 'DenseNet121'

with open('dense_model.json', "r") as json_file:
    loaded_model_json = json_file.read()

loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.load_weights('dense_weights.h5')
dense_model = loaded_model

predictions = dense_model.predict(test_ds)
confusion_matrix = tf.math.confusion_matrix(labels=predictions, predictions=predictions)

# Normalize confusion matrix to percentage
cm_percentage = confusion_matrix / tf.math.reduce_sum(confusion_matrix, axis=1, keepdims=True) * 100

# Create a dataframe for the heatmap
heatmap_df = pd.DataFrame(cm_percentage, columns=y_values, index=y_values)

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

fig.update_layout(coloraxis=dict(colorscale='Blues'), title_text='{} : {:.2f}%'.format('Dense Net 121', model_accuracy * 100))
fig.show() 