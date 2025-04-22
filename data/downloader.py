import kagglehub
import os
import shutil


PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))

class Downloader:
  def __init__(self, kaggle_url, dataset_dir):
    self.kaggle_url = kaggle_url
    self.dataset_dir = "{PROJECT_PATH}/{dataset_dir}".format(dataset_dir=dataset_dir)
    
  def download(self):
    path = kagglehub.dataset_download(self.kaggle_url)
    shutil.move("{path}/{dataset_path}".format(path=path), self.dataset_dir)
    print("Download Method Finished!")
