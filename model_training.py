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

img_height = 224
img_width = 224
epochs = 1

# test size is set to 5% of the dataset
#train, test = train_test_split(df_filtered_geo_data, test_size=0.05, random_state=123)

# Create training and test datasets using image_dataset_from_directory
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
test_ds = scaled_data.skip(train_size+train_size).take(test_size)

# number of classes = 2 - two continents
num_classes = len(loaded_data.class_names)

# Use TensorFlow's AUTOTUNE to automatically adjust the number of parallel calls during data preprocessing
AUTOTUNE = tf.data.AUTOTUNE

# Cache, shuffle, and prefetch the training dataset for optimized performance
train_ds_cache = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds_cache = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

dense_net = DenseNet121(
    input_shape=(img_height, img_width, 3),
    include_top=True,
    weights="imagenet",
    classifier_activation="softmax"
)

dense_model = Sequential()
dense_model.add(dense_net)
#dense_model.add(GlobalAveragePooling2D())
dense_model.add(Dense(num_classes, activation="softmax"))
dense_model.summary()

dense_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

dense_history = dense_model.fit(
    train_ds_cache,
    epochs=epochs,
    validation_data=val_ds_cache
)


# serialize model to JSON
dense_model_json = dense_model.to_json()
timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_name = "dense-" + timestr
with open(model_name, "w") as json_file:
    json_file.write(dense_model_json + ".json")
# serialize weights to HDF5
dense_model.save_weights(model_name+ ".h5")
print("Saved model to disk")

""" # plots
from plotly.subplots import make_subplots

epochs_range = [x for x in range(epochs)]

# Create a Plotly figure with subplots for accuracy and loss
fig = make_subplots(
    rows=2, 
    cols=2, 
    subplot_titles=("Training Accuracy",  "Validation Accuracy", "Training Loss", "Validation Loss")
)

plot_color = ['rgb(0, 0, 255)', 'rgb(255, 0, 0)', 'rgb(124,252,0)']

iter = 0
model_name = 'DenseNet121'
history = dense_history.history
    
# Extract training history values
acc, val_acc = history['accuracy'], history['val_accuracy']
loss, val_loss = history['loss'], history['val_loss']

# Add traces for accuracy
fig.add_trace(go.Scatter(x=epochs_range, y=acc, mode='lines', name=model_name, legendgroup=model_name, line_color=plot_color[iter]), row=1, col=1)
fig.add_trace(go.Scatter(x=epochs_range, y=val_acc, mode='lines', legendgroup=model_name, showlegend=False, line_color=plot_color[iter]), row=1, col=2)

# Add traces for loss
fig.add_trace(go.Scatter(x=epochs_range, y=loss, mode='lines', legendgroup=model_name, showlegend=False, line_color=plot_color[iter]), row=2, col=1)
fig.add_trace(go.Scatter(x=epochs_range, y=val_loss, mode='lines', legendgroup=model_name, showlegend=False, line_color=plot_color[iter]), row=2, col=2)


# Update layout for better readability
fig.update_layout(
    title_text='Training History',
    legend=dict(traceorder='normal', orientation='h', y=1.15, x=0.75, xref='paper', yref='paper'),
)

# Update y-axes titles and range for all subplots
for col in [1, 2]:
    fig.update_xaxes(title_text='Epochs', range=[0, epochs], row= col % 2 + 1, col=col)
    fig.update_xaxes(title_text='Epochs', range=[0, epochs], row= col, col=col)
    
    fig.update_yaxes(title_text='Accuracy', range=[0, 1], row=1, col=col)
    fig.update_yaxes(title_text='Loss', range=[0, 25], row=2, col=col)
    
fig.update_layout(height=800, width=1000)
#fig = add_plot_styling(fig)

# Show the plot
fig.show()

# get the true labels
true_labels = tf.concat([y for x, y in test_ds], axis=0)

# Generate predictions
predictions = dense_model.predict(test_ds)

# get the predicted labels
predicted_labels = tf.argmax(predictions, axis=1)

model_accuracy = np.mean(predicted_labels.numpy() == true_labels.numpy())

# Create confusion matrix
confusion_matrix = tf.math.confusion_matrix(
    labels=true_labels,
    predictions=predicted_labels,
    num_classes=len(test_ds.class_names)
)

# Normalize confusion matrix to percentage
cm_percentage = confusion_matrix / tf.math.reduce_sum(confusion_matrix, axis=1, keepdims=True) * 100

# Create a dataframe for the heatmap
heatmap_df = pd.DataFrame(cm_percentage, columns=test_ds.class_names, index=test_ds.class_names)

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
fig.show() """