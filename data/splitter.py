import fnmatch
import os
import shutil

import shared.classifications as classifications
import shared.paths as paths

classifications.init()
paths.init()

SPLIT_RATIO = 0.7

class Splitter:
  def split():
    unsorted_path = f"{paths.project_path}/dataset"
    for breed in classifications.breeds:
      class_path = f"{unsorted_path}/{breed}"
      filenames = fnmatch.filter(os.listdir(class_path), '*.jpg')
      split_index = round(len(filenames) * SPLIT_RATIO)

      print(f"Class Path:     {class_path}")
      print(f"Classification: {breed}")

      for index, filename in enumerate(filenames):
        target_dir = f"{paths.project_path}/training_dataset/{breed}"
        if index > split_index:
          target_dir = f"{paths.project_path}/validation_dataset/{breed}"

        os.makedirs(f"{target_dir}", exist_ok=True)
        shutil.copy(f"{unsorted_path}/{breed}/{filename}", f"{target_dir}/{filename}")

    # for file in os.listdir(path):
    #   if fnmatch

  def count_files(path):
    return len(fnmatch.filter(os.listdir(path), '*.jpg'))
