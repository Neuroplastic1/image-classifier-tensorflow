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

# Define the function to make prediction and return human-readable labels
def predict_and_label(model, image_path, class_names):
    """
    Imports an image located at image_path, makes a prediction with model
    and labels the image based on the highest probability prediction class.
    """
    # Load and prep the image
    img = load_and_prep_image(image_path)
    
    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))
    
    # Get the predicted class
    pred_class = class_names[int(tf.round(pred))]
    
    return pred_class

# Load class names for ImageNet
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

# Make a prediction
image_path = 'path_to_your_image.jpg'
prediction = predict_and_label(model, image_path, imagenet_labels)
print(prediction)

