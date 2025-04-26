import tensorflow as tf
from keras import layers
from keras import Sequential
import matplotlib.pyplot as plt
import numpy as np
import math

from prediction_helper import PredictionHelper
from data.downloader import Downloader
from data.splitter import Splitter

import shared.paths as paths
paths.init()

# %%
# Setting up the dataset

# Download dataset
KAGGLE_DATASET_URL = "khushikhushikhushi/dog-breed-image-dataset"
SPLIT_RATIO = 0.7

Downloader.download(KAGGLE_DATASET_URL)

# Split the dataset to its train / validate directories
# This would allow tf's image_dataset_from_driectory to properly use it
Splitter.split(SPLIT_RATIO)

# %%
# Creating Tensorflow Datasets
# NOTE: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory

training_directory = f"{paths.project_path}/training_dataset" 
validation_directory = f"{paths.project_path}/validation_dataset" 

IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
IS_SHUFFLE = True
BATCH_SIZE = 16

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(training_directory,
                                                            shuffle=IS_SHUFFLE,
                                                            image_size=IMG_SIZE,
                                                            batch_size=BATCH_SIZE)
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(validation_directory,
                                                            shuffle=IS_SHUFFLE,
                                                            image_size=IMG_SIZE,
                                                            batch_size=BATCH_SIZE)
class_names = train_dataset.class_names
number_of_classes = len(train_dataset.class_names)
# Visualise the training dataset in a grid of images
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(0, 15):
    ax = plt.subplot(5, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# %% 
# Data Preprocessing

# Create a test dataset from the validation dataset
# NOTE: https://www.tensorflow.org/api_docs/python/tf/data/experimental/cardinality
valdation_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(valdation_batches // 5)
validation_dataset = validation_dataset.skip(valdation_batches // 5)

print(f"Number of validation batches: {tf.data.experimental.cardinality(validation_dataset)}")
print(f"Number of test batches: {tf.data.experimental.cardinality(test_dataset)}")

# Data Augmentation
# NOTE: https://www.tensorflow.org/tutorials/images/data_augmentation#data_augmentation_2

data_augmentation = Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

# Visualize the Augmented Data
for image, _ in train_dataset.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(0, 15):
    ax = plt.subplot(5, 3, i + 1)
    augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')


# %%
# Create Base model using Resnet50
# https://www.tensorflow.org/tutorials/images/transfer_learning#create_the_base_model_from_the_pre-trained_convnets
# https://medium.com/@nitishkundu1993/exploring-resnet50-an-in-depth-look-at-the-model-architecture-and-code-implementation-d8d8fa67e46f
# https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50?hl=en

# Pretrained has 176 layers
pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                                                  pooling='avg',
                                                  weights='imagenet')

pretrained_model.trainable = False

# %%
# Resnet Preprocessing
# https://medium.com/@kenneth.ca95/a-guide-to-transfer-learning-with-keras-using-resnet50-a81a4a28084b
# https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet_v2/preprocess_input
# https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub#simple_transfer_learning

preprocess_input = tf.keras.applications.resnet50.preprocess_input

# %%
# Data Prefetching
# NOTE: https://www.tensorflow.org/guide/data_performance
AUTOTUNE = tf.data.AUTOTUNE # -1

# Buffer_size: If the value tf.data.AUTOTUNE is used, then the buffer size is dynamically tuned. 
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# %%
# Classification
# https://www.tensorflow.org/tutorials/images/transfer_learning#add_a_classification_head
# https://medium.com/@kenneth.ca95/a-guide-to-transfer-learning-with-keras-using-resnet50-a81a4a28084b
# https://stackoverflow.com/questions/65609285/in-tensorflow-adding-data-augmentation-layers-to-my-keras-model-slows-down-trai

input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

# resnet_model = Sequential([
#   layers.InputLayer(shape=input_shape),
#   data_augmentation,
#   pretrained_model,
#   layers.Flatten(),
#   layers.BatchNormalization(),
#   layers.Dense(1024, activation='relu'),
#   layers.Dropout(0.2),
#   layers.BatchNormalization(),
#   layers.Dense(512, activation='relu'),
#   layers.Dropout(0.2),
#   layers.BatchNormalization(),
#   layers.Dense(256, activation='relu'),
#   layers.Dropout(0.2),
#   layers.BatchNormalization(),
#   layers.Dense(number_of_classes, activation='softmax')
# ])

prediction_layer = layers.Dense(number_of_classes, activation='softmax')

inputs = layers.Input(shape=input_shape)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = pretrained_model(x, training=False)
x = layers.Flatten()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.BatchNormalization()(x)
outputs = prediction_layer(x)

resnet_model = tf.keras.Model(inputs, outputs)


# %%
# Compiling and Training
# https://medium.com/@nitishkundu1993/exploring-resnet50-an-in-depth-look-at-the-model-architecture-and-code-implementation-d8d8fa67e46f
# https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
# https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CategoricalAccuracy
# https://www.geeksforgeeks.org/categorical-cross-entropy-in-multi-class-classification/

initial_epochs = 15
base_learning_rate = 0.001
resnet_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                       metrics=['accuracy'])

resnet_model.summary()
tf.keras.utils.plot_model(resnet_model, show_shapes=True)

history = resnet_model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)


# %%
# Learning Curves
# https://www.tensorflow.org/tutorials/images/transfer_learning#

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# %%
# Mass Prediction w/o Fine tuning

sample_set = {
  "Beagle": "https://i.pinimg.com/736x/51/c6/0f/51c60fcb12a5fdfb2af386d296f5d14e.jpg",
  "Boxer": "https://i.pinimg.com/736x/2e/a7/ce/2ea7cea704052ae3673fcd33118c1db4.jpg",
  "Bulldog": "https://i.pinimg.com/736x/71/42/fb/7142fbd8f9467b9a09d982d0a197ad61.jpg",
  "Hotdog_Dog": "https://i.pinimg.com/736x/bc/14/f6/bc14f673187a0b67ba7f940ef30ef392.jpg",
  "German_Shepherd": "https://i.pinimg.com/736x/1b/96/8f/1b968f0c5076f5fd840cc37d8770207a.jpg",
  "Golden_Retriever": "https://i.pinimg.com/736x/1c/80/7e/1c807eb83d6e24c7dfe03d8bc97c43bf.jpg",
  "Labrador": "https://i.pinimg.com/736x/14/e3/a2/14e3a216753fd795848cc22417b27e40.jpg",
  "Poodle": "https://i.pinimg.com/736x/3f/23/43/3f2343558d24c0bd0836e2466961d7a6.jpg",
  "Rotty": "https://i.pinimg.com/736x/c5/d7/a3/c5d7a37654da3ebe930b3fb721f31181.jpg",
  "Yorkie": "https://i.pinimg.com/736x/45/96/db/4596db77a4ac82c5fcd709a7c0ec6d85.jpg"
}

for key in sample_set:
  print("==============================================================")
  print(f"Actual Breed: {key}")
  p_helper = PredictionHelper(resnet_model, class_names, (IMG_HEIGHT, IMG_WIDTH))
  p_helper.predict(sample_set[key])

# %%
# Mass prediction with test dataset w/o Fine tuning
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = resnet_model.predict_on_batch(image_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
  score = tf.nn.softmax(predictions[i])
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(f"{class_names[np.argmax(score)]}, conf: {math.floor(100 * np.max(score))}")
  plt.axis("off")

# %%
# Fine Tuning

fine_tune_epochs = 15
total_epochs =  initial_epochs + fine_tune_epochs
pretrained_model.trainable = True
fine_tune_at = 150

for layer in pretrained_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile the 
resnet_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                       metrics=['accuracy'])

resnet_model.summary()

history_fine = resnet_model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=len(history.epoch),
                         validation_data=validation_dataset)

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# %%
# Mass Prediction

sample_set = {
  "Beagle": "https://i.pinimg.com/736x/51/c6/0f/51c60fcb12a5fdfb2af386d296f5d14e.jpg",
  "Boxer": "https://i.pinimg.com/736x/2e/a7/ce/2ea7cea704052ae3673fcd33118c1db4.jpg",
  "Bulldog": "https://i.pinimg.com/736x/71/42/fb/7142fbd8f9467b9a09d982d0a197ad61.jpg",
  "Hotdog_Dog": "https://i.pinimg.com/736x/bc/14/f6/bc14f673187a0b67ba7f940ef30ef392.jpg",
  "German_Shepherd": "https://i.pinimg.com/736x/1b/96/8f/1b968f0c5076f5fd840cc37d8770207a.jpg",
  "Golden_Retriever": "https://i.pinimg.com/736x/1c/80/7e/1c807eb83d6e24c7dfe03d8bc97c43bf.jpg",
  "Labrador": "https://i.pinimg.com/736x/14/e3/a2/14e3a216753fd795848cc22417b27e40.jpg",
  "Poodle": "https://i.pinimg.com/736x/3f/23/43/3f2343558d24c0bd0836e2466961d7a6.jpg",
  "Rotty": "https://i.pinimg.com/736x/c5/d7/a3/c5d7a37654da3ebe930b3fb721f31181.jpg",
  "Yorkie": "https://i.pinimg.com/736x/45/96/db/4596db77a4ac82c5fcd709a7c0ec6d85.jpg"
}

for key in sample_set:
  print("==============================================================")
  print(f"Actual Breed: {key}")
  p_helper = PredictionHelper(resnet_model, class_names, (IMG_HEIGHT, IMG_WIDTH))
  p_helper.predict(sample_set[key])

# %%
# Mass prediction with test dataset
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = resnet_model.predict_on_batch(image_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
  score = tf.nn.softmax(predictions[i])
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(f"{class_names[np.argmax(score)]}, conf: {math.floor(100 * np.max(score))}")
  plt.axis("off")


# %%
