#!/usr/bin/env python
# coding: utf-8

# In[12]:


import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

# Define the URL for the dataset
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

# Download and extract the dataset
dataset_dir = tf.keras.utils.get_file('flower_photos', dataset_url, untar=True, cache_dir='.', cache_subdir='')

# Load the dataset
batch_size = 32
img_height = 180
img_width = 180

train_ds = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

from tensorflow.keras import layers, models

num_classes = len(train_ds.class_names)

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

model.save('flower_classification_model.h5')

import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import PIL

# Load the trained model
model = load_model('flower_classification_model.h5')

# Load class names
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

def predict_image(img):
    img = img.resize((180, 180))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return class_names[np.argmax(score)]

# Streamlit UI
st.title("Flower Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    label = predict_image(image)
    st.write(f"This flower is a: **{label}**")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




