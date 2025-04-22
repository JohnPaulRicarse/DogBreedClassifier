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
      print(class_path)

      filenames = fnmatch.filter(os.listdir(class_path), '*.jpg')
      split_index = round(len(filenames) * SPLIT_RATIO)
      print("Split index", split_index)

      target_dir = f"{paths.project_path}/sorted_dataset/{breed}/"
      print("Target Dir: ", target_dir)
      for index, filename in enumerate(filenames):
        subfolder = 'train'
        if index > split_index:
          subfolder = 'validation'

        os.makedirs(f"{target_dir}/{subfolder}", exist_ok=True)
        shutil.copy(f"{unsorted_path}/{breed}/{filename}", f"{target_dir}/{subfolder}/{filename}")

    # for file in os.listdir(path):
    #   if fnmatch

  def count_files(path):
    return len(fnmatch.filter(os.listdir(path), '*.jpg'))
