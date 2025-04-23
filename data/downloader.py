import kagglehub
import shutil

import shared.paths as paths

paths.init()

class Downloader:
  def download(kaggle_url):
    path = kagglehub.dataset_download(kaggle_url)
    print(f"Path is: {path}")
    shutil.copytree(f"{path}", paths.project_path, dirs_exist_ok=True)
    print("Download Method Finished!")
