from data.downloader import Downloader
from data.splitter import Splitter

# %%
# %reset -f
# Download Datasets
KAGGLE_DATASET_URL = "khushikhushikhushi/dog-breed-image-dataset"
DATASET_PATH = "/dataset"

# downloader = Downloader(KAGGLE_DATASET_URL, DATASET_PATH)
# downloader.download()

# %%
# Split! 🪓
Splitter.split()
