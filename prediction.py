import os
import tensorflow as tf
import numpy as np

class Prediction:
  def __init__(self, resnet_model, class_names, img_shape):
    self.resnet_model = resnet_model
    self.class_names = class_names
    self.img_shape = img_shape
    
  def predict(self, picture_url):
    dog_pic_path = tf.keras.utils.get_file('dogbreed', origin=picture_url)

    img = tf.keras.utils.load_img(
        dog_pic_path, target_size=self.img_shape
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = self.resnet_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(self.class_names[np.argmax(score)], 100 * np.max(score))
    )

    os.remove(dog_pic_path)

