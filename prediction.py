import os
import tensorflow as tf
import numpy as np

class Prediction:
  def predict(picture_url, model, class_names, img_shape):
    dog_pic_path = tf.keras.utils.get_file('dogbreed', origin=picture_url)

    img = tf.keras.utils.load_img(
        dog_pic_path, target_size=img_shape
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

    os.remove(dog_pic_path)

