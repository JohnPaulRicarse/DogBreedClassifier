from data.downloader import Downloader
from data.splitter import Splitter

# %%
# Setting up the dataset

# Download dataset
KAGGLE_DATASET_URL = "khushikhushikhushi/dog-breed-image-dataset"
Downloader.download(KAGGLE_DATASET_URL)

# Split the dataset to its train / validate directories
# This would allow tf's image_dataset_from_driectory to properly use it
Splitter.split()

# %%
# Feeding Datasets to Tensorflow
# NOTE: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory

