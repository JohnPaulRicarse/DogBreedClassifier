from data.downloader import Downloader
from data.splitter import Splitter

# %%
# Setting up the dataset

# Download dataset
KAGGLE_DATASET_URL = "khushikhushikhushi/dog-breed-image-dataset"
Downloader.download(KAGGLE_DATASET_URL)

# Split the dataset to its train / validate directories
Splitter.split()

