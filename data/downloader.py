import kagglehub
import shutil
import shared.paths as paths
paths.init()

class Downloader:
  def __init__(self, kaggle_url, dataset_dir):
    self.kaggle_url = kaggle_url
    self.dataset_dir = "{project_path}".format(dataset_dir=dataset_dir, project_path=paths.project_path)
    
  def download(self):
    path = kagglehub.dataset_download(self.kaggle_url)
    shutil.move("{path}/dataset".format(path=path), self.dataset_dir)
    print("Path is: {path}".format(path=path))
    print("Dataset Path is: {path}".format(path=self.dataset_dir))
    print("Download Method Finished!")
