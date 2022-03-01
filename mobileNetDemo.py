# %%
# Import necessary packages and Libraries
from IPython.display import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt

# %%
# Download mobilenet
mobile = tf.keras.applications.mobilenet.MobileNet()

# %%


def prepare_image(file):
    """
    Preparing and processing Images for MobileNet
    """
    img_path = 'data/MobileNet-samples/'
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


# %%

# -------------------- First Image---------------------
# Display the image to be processed, and to be predicted
Image(filename='data/MobileNet-samples/1.jpg', width=300, height=200)

# %%
# Process the image for MobileNet, by pasing the file name
preprocessed_image = prepare_image('1.jpg')

# Make predictions on the processed image
predictions = mobile.predict(preprocessed_image)

# Find the top 5 predictions
results = imagenet_utils.decode_predictions(predictions)

# Display the predictions
results

# %%
# ---------------------- Second Image----------------------
Image(filename='data/MobileNet-samples/panda.jpg', width=500, height=300)

# %%
preprocessed_image = prepare_image('panda.jpg')

predictions = mobile.predict(preprocessed_image)

results = imagenet_utils.decode_predictions(predictions)

results

# %%
