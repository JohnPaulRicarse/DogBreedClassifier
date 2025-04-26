import tensorflow as tf
from keras import layers
from keras import Sequential
import matplotlib.pyplot as plt

from prediction import Prediction
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

IMG_WIDTH = 160
IMG_HEIGHT = 100
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
IS_SHUFFLE = True
BATCH_SIZE = 32

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
# Create Base model using Resnet50
# https://www.tensorflow.org/tutorials/images/transfer_learning#create_the_base_model_from_the_pre-trained_convnets
# https://medium.com/@nitishkundu1993/exploring-resnet50-an-in-depth-look-at-the-model-architecture-and-code-implementation-d8d8fa67e46f
# https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50?hl=en

pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                                                  pooling='avg',
                                                  weights='imagenet')

for layer in pretrained_model.layers:
    layer.trainable = False

# %%
# Classification
# https://www.tensorflow.org/tutorials/images/transfer_learning#add_a_classification_head
# https://medium.com/@nitishkundu1993/exploring-resnet50-an-in-depth-look-at-the-model-architecture-and-code-implementation-d8d8fa67e46f
# https://stackoverflow.com/questions/65609285/in-tensorflow-adding-data-augmentation-layers-to-my-keras-model-slows-down-trai

input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
resnet_model = Sequential([
  layers.InputLayer(input_shape=input_shape),
  data_augmentation,
  pretrained_model,
  layers.Dense(1024, activation='relu'),
  layers.Dense(512, activation='relu'),
  layers.Dense(number_of_classes)
])

resnet_model.summary()

# %%
# Compiling and Training
# https://medium.com/@nitishkundu1993/exploring-resnet50-an-in-depth-look-at-the-model-architecture-and-code-implementation-d8d8fa67e46f
# https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
# https://www.tensorflow.org/api_docs/python/tf/keras/metrics/CategoricalAccuracy
# https://www.geeksforgeeks.org/categorical-cross-entropy-in-multi-class-classification/

#  In loss categorical_crossentropy, expected y_pred.shape to be (batch_size, num_classes) with num_classes > 1. Received: y_pred.shape=(None, 1). Consider using 'binary_crossentropy' if you only have 2 classes.
# return self.fn(y_true, y_pred, **self._fn_kwargs)

initial_epochs = 10
base_learning_rate = 0.001
resnet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'])

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
# Predictions
dog_pic_url = "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2020/07/28113003/Yorkshire-Terrier-puppy-in-a-dog-bed.20200601164413905.jpg"
Prediction.predict(dog_pic_url, resnet_model, class_names, (IMG_HEIGHT, IMG_WIDTH))
