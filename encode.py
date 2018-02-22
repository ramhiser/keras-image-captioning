from keras.applications import VGG16
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import Model

import numpy as np

import argparse
import sys
import logging


def load_encoding_model():
    """Model to encode image as vector of length 4096 using 2nd to last layer of
    VGG16"""
    base_model = VGG16(weights='imagenet', include_top=True)
    encoding_model = Model(inputs=base_model.input,
                           outputs=base_model.get_layer('fc2').output)

    return encoding_model


def load_image(image_filename, input_shape=None):
    if input_shape is None:
        input_shape = (224, 224)
    image = load_img(image_filename, target_size=input_shape)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    return image


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Encode images using VGG16")
    parser.add_argument("-i", "--image", required=True,
                        help="path to the input image")

    args = parser.parse_args(sys.argv[1:])

    encoding_model = load_encoding_model()

    image = load_image(args.image)
    image_encoded = encoding_model.predict(image)

    print(image_encoded)
    print(image_encoded.shape)
