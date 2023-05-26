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
