import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
# Load Keras dependencies:
import tensorflow as tf
from tensorflow.keras.preprocessing import image
tf.keras.backend.set_image_data_format('channels_first')

import glob

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def CheckDirectory(path):
    if not os.path.exists(path):
        pathlib.Path(path).mkdir()

def read_preprocess_image(imgPath, img_width, img_height):
    img = load_img(imgPath, target_size=(img_width, img_height))
    imgArray = img_to_array(img)
    imgArray = imgArray.reshape(1, 3, img_width, img_height)
    imgArray = imgArray / float(255)
    return imgArray

def image_reshape(image):
    shape = np.shape(image)
    if len(shape) == 4:
        return image[0]
    elif len(shape) == 3:
        return np.expand_dims(x, axis=0)

def moveaxis_change_channel_order(image):
    shape = np.shape(image)
    if len(shape) == 4:
        if shape[1] == 3:
            return np.moveaxis(image, 1, -1)
        elif shape[-1] == 3:
            return np.moveaxis(image, -1, 1)
    elif len(shape) == 3:
        if shape[0] == 3:
            return np.moveaxis(image, 0, -1)
        elif shape[-1] == 3:
            return np.moveaxis(image, -1, 0)
    return image

#attack label: 0
#real label: 1

def get_spoofying_image():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs, ", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    model_path = 'REPLAY-HalfFPS_val96.h5'
    model = load_model(model_path)
    img_width = 96
    img_height = 96
    read_image_path = 'real_choose_cropped'
    output_root = 'anti-spoofing test real image'
    CheckDirectory(output_root)
    extension = '.png'
    real_label = 1
    attack_label = 0

    for dname in os.listdir(read_image_path):
        if os.path.isdir(os.path.join(read_image_path, dname)):
            CheckDirectory(os.path.join(output_root, dname))
            for img_name in os.listdir(os.path.join(read_image_path, dname)):
                image = read_preprocess_image(os.path.join(read_image_path, dname, str(img_name)), img_width, img_height)
                #print("pre ", model.predict(image))
                #print("label ", model.predict(image) > 0.5)
                label = (model.predict(image) > 0.5)[0][0].astype("int32")
                if label == real_label:
                    plt.imsave(os.path.join(output_root, dname, str(img_name)), moveaxis_change_channel_order(image_reshape(image)))

if __name__ == "__main__":
    get_spoofying_image()
