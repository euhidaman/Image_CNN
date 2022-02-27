# %%
from tabnanny import verbose
import numpy as np
from sklearn.model_selection import validation_curve
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# %%
# This code is used to organize the data.
# But, it's throwing a large dataset-error, so no need to run it
os.chdir('data/dogs-vs-cats')
if os.path.isdir('train/dog') is False:
    os.makedirs('train/dog')
    os.makedirs('train/cat')
    os.makedirs('valid/dog')
    os.makedirs('valid/cat')
    os.makedirs('test/dog')
    os.makedirs('test/cat')

    for c in random.sample(glob.glob('cat*'), 500):
        shutil.move(c, 'train/cat')
    for c in random.sample(glob.glob('dog*'), 500):
        shutil.move(c, 'train/dog')
    for c in random.sample(glob.glob('cat*'), 100):
        shutil.move(c, 'valid/cat')
    for c in random.sample(glob.glob('dog*'), 100):
        shutil.move(c, 'valid/dog')
    for c in random.sample(glob.glob('cat*'), 50):
        shutil.move(c, 'test/cat')
    for c in random.sample(glob.glob('dog*'), 50):
        shutil.move(c, 'test/dog')

# %%
os.chdir('../../')
os.getcwd()

# %%
# Path to various datasets locations
train_path = 'data/dogs-vs-cats/train'
valid_path = 'data/dogs-vs-cats/valid'
test_path = 'data/dogs-vs-cats/test'

# %%
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10, shuffle=False)

# %%
# Verifying correct no. of images were found in the correct place
assert train_batches.n == 1000
assert valid_batches.n == 200
assert test_batches.n == 100
assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 2

# %%
# Grabbing a single block of images and corresponding labels, from our training batch
imgs, labels = next(train_batches)

# %%

# Function plots images in a grid of 1 row and 10 columns
# Below code, is directly taken from Tensorflow's documentation


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# %%
plotImages(imgs)
print(labels)

# %%

# Creating the model
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
           padding='same', input_shape=(224, 224, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax'),
])

model.summary()

# %%

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)

# %%
# ------------------Predicting labels for Test Images----------------------

# Taking a batch of test-data, and predicting for it
test_imgs, test_labels = next(test_batches)
plotImages(test_imgs)
print(test_labels)

# %%
test_batches.classes
# %%
predictions = model.predict(x=test_batches, verbose=0)
# %%
np.round(predictions)

# %%
# Plotting the Confusion Matrix
cm = confusion_matrix(y_true=test_batches.classes,
                      y_pred=np.argmax(predictions, axis=-1))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# %%
test_batches.class_indices

# %%
cm_plot_labels = ['cat', 'dog']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

# %%
# --------------------Building Fine-Tuned VGG16 Model---------------------------

# Download Model from Internet
vgg16_model = tf.keras.applications.vgg16.VGG16()

# %%
vgg16_model.summary()

# %%

# The VGG16 model is a functional model
# So, we need to convert it into a Sequential model.
# And, that's why we do the below thing,
# which puts all the model values from start to second-last.
model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

# %%

# This code, freezes the trainable parameters of the new model.
# Reason - We don't want to train it again, so as to not update the weights in the model.
# Bcz, VGG16 has already pre-learnt the fetures of Cats & Dogs
for layer in model.layers:
    layer.trainable = False

# Manually, add the last layer of 2 nodes
# (bcz, we are trying to find, whether an image belongs to the classes of : Cats/Dogs)
model.add(Dense(units=2, activation='softmax'))
model.summary()

# %%
# Compiling and Training the modified VGG16 model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches, validation_data=valid_batches, epochs=5, verbose=2)

# %%

# Predict using new model
predictions = model.predict(x=test_batches, verbose=0)

test_batches.classes

# %%
# Plotting the Confusion Matrix
cm = confusion_matrix(y_true=test_batches.classes,
                      y_pred=np.argmax(predictions, axis=-1))

# %%
test_batches.class_indices

# %%
cm_plot_labels = ['cat', 'dog']

plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

# %%
