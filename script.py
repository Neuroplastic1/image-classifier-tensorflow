import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

# Load pre-trained MobileNetV2 model from TensorFlow Hub
model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", 
                   output_shape=[1001],
                   trainable=False)
])

# Define the function to load and preprocess images
def load_and_prep_image(image_path, img_shape=224):
    """
    Reads an image from filename, turns it into a tensor and reshapes it
    to img_shape (img_shape, img_shape, color_channels)
    """
    # Read in the image
    img = tf.io.read_file(image_path)
    # Decode the read file into a tensor
    img = tf.image.decode_image(img)
    # Resize the image  
    img = tf.image.resize(img, size=[img_shape, img_shape])
    # Rescale the image (get all values between 0 and 1)
    img = img/255.
    return img
