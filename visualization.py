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

true_labels = np.concatenate(list(test_ds_labels.map(lambda x, y: y).as_numpy_iterator()))
print(true_labels)

iter = 0
model_name = 'DenseNet121'

with open('dense_model.json', "r") as json_file:
    loaded_model_json = json_file.read()

loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.load_weights('dense_weights.h5')
dense_model = loaded_model

true_labels = np.concatenate(list(test_ds.map(lambda x, y: y).as_numpy_iterator()))
true_labels_int = true_labels

# Get model predictions
predictions = dense_model.predict(test_ds)

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

class_names = ['Asia', 'Europe']
heatmap_df = pd.DataFrame(cm_percentage.numpy(), index=class_names, columns=class_names)

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