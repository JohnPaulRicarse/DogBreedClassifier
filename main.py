import tensorflow as tf
from keras import layers
from keras import Sequential
import matplotlib.pyplot as plt

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

IMG_SIZE = (100, 160)
IS_SHUFFLE = True
BATCH_SIZE = 15

train_dataset = tf.keras.utils.image_dataset_from_directory(training_directory,
                                                            shuffle=IS_SHUFFLE,
                                                            image_size=IMG_SIZE,
                                                            batch_size=BATCH_SIZE)
validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_directory,
                                                            shuffle=IS_SHUFFLE,
                                                            image_size=IMG_SIZE,
                                                            batch_size=BATCH_SIZE)
class_names = train_dataset.class_names

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

# Prefetch the images into memory for performance
# NOTE: https://www.tensorflow.org/guide/data_performance
AUTOTUNE = tf.data.AUTOTUNE # -1

# Buffer_size: If the value tf.data.AUTOTUNE is used, then the buffer size is dynamically tuned. 
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

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
