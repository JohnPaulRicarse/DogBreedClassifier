from data.downloader import Downloader

KAGGLE_DATASET_URL = "khushikhushikhushi/dog-breed-image-dataset"
DATASET_PATH = "/dataset"

downloader = Downloader()
downloader.download(KAGGLE_DATASET_URL, DATASET_PATH)
