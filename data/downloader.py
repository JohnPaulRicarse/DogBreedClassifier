import kagglehub
import shutil

import shared.paths as paths

paths.init()

# /home/jp/.cache/kagglehub/datasets/khushikhushikhushi/dog-breed-image-dataset/versions/1
class Downloader:
  def download(kaggle_url):
    path = kagglehub.dataset_download(kaggle_url)
    print(f"Path is: {path}")
    shutil.copytree(f"{path}", paths.project_path, dirs_exist_ok=True)
    print("Download Method Finished!")
