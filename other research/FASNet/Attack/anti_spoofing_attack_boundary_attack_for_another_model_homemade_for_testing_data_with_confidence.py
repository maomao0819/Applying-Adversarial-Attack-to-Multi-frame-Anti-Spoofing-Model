import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import pathlib
from matplotlib import gridspec
import pickle
import time
import datetime
from PIL import Image
import json
import random
# Load Keras dependencies:
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow.keras
tensorflow.keras.backend.set_image_data_format('channels_first')

import glob

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def name_list_gen(data_dir):
    name_list = []
    im_path = data_dir + '*.png'
    for image_path in sorted(glob.glob(im_path)):
        temp_name = image_path[len(data_dir) : len(image_path) + 1]
        name_list.append(temp_name)
    return name_list

def read_preprocess_image(imgPath, img_width, img_height):
    img = load_img(imgPath, target_size=(img_width, img_height))
    imgArray = img_to_array(img)
    imgArray = imgArray.reshape(1, 3, img_width, img_height)
    imgArray = imgArray / float(255)
    return imgArray

#attack label: 0
#real label: 1

def label_confidence_image(image, model):
    image_copy = image
    image_probs = model.predict(image_copy)
    outLabel = (image_probs > 0.5).astype("int32")
    # pred_class = np.argmax(image_probs, axis=-1)
    label = outLabel
    confidence = np.amax(image_probs)
    img = np.array(image_copy)
    img = np.moveaxis(img, 1, -1)
    img = img[0]
    return label, confidence, img

def check_label(data_dir, name_list, model, img_width, img_height):
    left_bound = 64
    right_bound = 317
    for img_name in name_list:
        img_num = img_name.replace(".png", "")
        if int(img_num) >= left_bound and int(img_num) <= right_bound:
            image_dir = data_dir + img_name
            image = read_preprocess_image(image_dir, img_width, img_height)
            label = (model.predict(image) > 0.5)[0][0].astype("int32")
            if label == 0:
                print(img_num)

def CheckDirectory(path_save):
    if not os.path.exists(path_save):
        pathlib.Path(path_save).mkdir()

def orthogonal_perturbation(delta, prev_sample, target_sample):
    prev_sample = prev_sample.reshape(96, 96, 3)
    # Generate perturbation
    perturb = np.random.randn(96, 96, 3)
    perturb /= get_diff(perturb, np.zeros_like(perturb))
    perturb *= delta * np.mean(get_diff(target_sample, prev_sample))
    # Project perturbation onto sphere around target
    diff = (target_sample - prev_sample).astype(np.float32)
    diff /= get_diff(target_sample, prev_sample)
    diff = diff.reshape(3, 96, 96)
    perturb = perturb.reshape(3, 96, 96)
    for i, channel in enumerate(diff):
        perturb[i] -= np.dot(perturb[i], channel) * channel
    # Check overflow and underflow
    mean = [0, 0, 0]
    perturb = perturb.reshape(96, 96, 3)
    overflow = (prev_sample + perturb) - np.concatenate((np.ones((96, 96, 1)) * (255. - mean[0]), np.ones((96, 96, 1)) * (255. - mean[1]), np.ones((96, 96, 1)) * (255. - mean[2])), axis=2)
    overflow = overflow.reshape(96, 96, 3)
    perturb -= overflow * (overflow > 0)
    underflow = np.concatenate((np.ones((96, 96, 1)) * (0. - mean[0]), np.ones((96, 96, 1)) * (0. - mean[1]), np.ones((96, 96, 1)) * (0. - mean[2])), axis=2) - (prev_sample + perturb)
    underflow = underflow.reshape(96, 96, 3)
    perturb += underflow * (underflow > 0)
    return perturb

def forward_perturbation(epsilon, prev_sample, target_sample):
    perturb = (target_sample - prev_sample).astype(np.float32)
    perturb /= get_diff(target_sample, prev_sample)
    perturb *= epsilon
    return perturb

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

def draw(sample, model, path):
    label, confidence, img = label_confidence_image(np.copy(sample), model)
    sample = moveaxis_change_channel_order(sample)
    sample = sample.reshape(96, 96, 3)
    # Reverse preprocessing, see https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py
    sample *= 255
    sample = sample.astype(np.uint8)
    # Convert array to image and save
    sample = Image.fromarray(sample)
    id_no = time.strftime('%Y%m%d_%H%M%S', datetime.datetime.now().timetuple())
    # Save with predicted label for image (may not be adversarial due to uint8 conversion)
    label = label[0][0]
    if label == 1:
        label = "real"
    elif label == 0:
        label = "attack"
    sample.save(path)

def get_diff(sample_1, sample_2):
    sample_1 = sample_1.reshape(3, 96, 96)
    sample_2 = sample_2.reshape(3, 96, 96)
    diff = []
    for i, channel in enumerate(sample_1):
        diff.append(np.linalg.norm((channel - sample_2[i]).astype(np.float32)))
    return np.array(diff)

def boundary_attack():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs, ", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    # model_path = 'REPLAY-HalfFPS_val96.h5'
    model_path = 'REPLAY_homemade_776_half_epoch2.h5'
    if model_path:
        model = load_model(model_path)
    else:
        print('Could not load model!')

    model.summary()

    img_width, img_height = (96, 96)
    # path_save = "adversarial_example_boundary_attack/"
    path_save = "adversarial_example_boundary_attack_for_testing_data_test_100_124_confidence_888"
    # path_read_attack_image = "anti-spoofing test attack image/"
    path_read_attack_image = 'homemade_attack_choose_cropped_clients_test_100_124'
    # path_read_real_image = "anti-spoofing test real image/"
    path_read_real_image = 'real_image_cropped'
    CheckDirectory(path_save)
    for dname in os.listdir(path_read_attack_image):
        if os.path.isdir(os.path.join(path_read_attack_image, dname)):
            print("dname: ", dname)
            # for image_type in os.listdir(os.path.join(path_read_attack_image, dname)):
            #     if os.path.isdir(os.path.join(path_read_attack_image, dname, image_type)):
            dname_list = dname.split('_')
            if int(dname_list[2]) > 100:
                strange_attack_fail = 0
                attack_success = 0
                attack_fail = 0
                CheckDirectory(os.path.join(path_save, dname))
                file_num = 0
                for _ in os.listdir(os.path.join(path_save, dname)):
                    file_num += 1
                # for i in range(130 - file_num):
                for img_name in os.listdir(os.path.join(path_read_attack_image, dname)):
                    # img_name = random.choice(os.listdir(os.path.join(path_read_attack_image, dname))) 
                    is_exists = False
                    for image_exist in os.listdir(os.path.join(path_save, dname)):
                        if img_name == image_exist.split('_')[6]:
                            is_exists = True
                            break
                    if is_exists:
                        continue
                    is_strange_attack_fail = False
                    target_image_dir = os.path.join(path_read_attack_image, dname, img_name)
                    print("target_image_dir", target_image_dir)
                    target_sample = read_preprocess_image(target_image_dir, img_width, img_height)
                    random_image = random.choice(os.listdir(os.path.join(path_read_real_image, f'client{dname_list[-2]}', dname_list[-1])))
                    initial_image_dir = os.path.join(path_read_real_image, f'client{dname_list[-2]}', dname_list[-1], random_image)
                    print("initial_image_dir", initial_image_dir)
                    initial_sample = read_preprocess_image(initial_image_dir, img_width, img_height)
                    img_num = img_name.replace(".png", "")

                    confidence_limit = 0.8
                    attack_class = (model.predict(initial_sample) > confidence_limit)[0][0].astype("int32")
                    print("attack_class", attack_class)
                    target_class = (model.predict(target_sample) > 0.5)[0][0].astype("int32")
                    print("target_class", target_class)
                    
                    # real: 1
                    # attack: 0
                    
                    while attack_class == 0:
                        random_image = random.choice(os.listdir(os.path.join(path_read_real_image, f'client{dname_list[-2]}', dname_list[-1])))
                        initial_image_dir = os.path.join(path_read_real_image, f'client{dname_list[-2]}', dname_list[-1], random_image)
                        print("initial_image_dir", initial_image_dir)
                        initial_sample = read_preprocess_image(initial_image_dir, img_width, img_height)
                        # print("attack_class", attack_class)
                        attack_class = (model.predict(initial_sample) > confidence_limit)[0][0].astype("int32")

                    adversarial_sample = initial_sample

                    if target_class == 1:
                        is_strange_attack_fail = True
                        strange_attack_fail += 1
                        target_class = 0
                        continue
                    n_steps = 0
                    n_calls = 0
                    epsilon = 1.
                    delta = 0.1
                    diff_threshold = 1e-3
                    step_limit = 1500
                    
                    # Move first step to the boundary
                    adversarial_sample = moveaxis_change_channel_order(adversarial_sample)
                    target_sample = moveaxis_change_channel_order(target_sample)
                    while True:
                        trial_sample = adversarial_sample + forward_perturbation(epsilon * get_diff(adversarial_sample, target_sample), adversarial_sample, target_sample)
                        prediction = model.predict(moveaxis_change_channel_order(trial_sample))
                        n_calls += 1
                        if (prediction > confidence_limit)[0][0].astype("int32") == attack_class:
                            adversarial_sample = trial_sample
                            break
                        else:
                            epsilon *= 0.9
                        # if n_calls >= 5000:
                            # break
                    # print("\tForward perturbation finish")
                    # print("predict ", model.predict(moveaxis_change_channel_order(adversarial_sample))[0][0])
                    while True:
                        print("Step #{}...".format(n_steps))
                        # print("\tDelta step...")
                        d_step = 0
                        while True:
                            d_step += 1
                            # print("\t#{}".format(d_step))
                            trial_samples = []
                            for i in np.arange(10):
                                trial_sample = adversarial_sample + orthogonal_perturbation(delta, adversarial_sample, target_sample)
                                trial_samples.append(trial_sample)
                            predictions = model.predict(moveaxis_change_channel_order(np.array(trial_samples).reshape(-1, 96, 96, 3)))
                            n_calls += 10
                            predictions = (predictions > confidence_limit).astype("int32")
                            d_score = np.mean(predictions == attack_class)
                            if d_score > 0.0:
                                if d_score < 0.3:
                                    delta *= 0.9
                                elif d_score > 0.7:
                                    delta /= 0.9
                                adversarial_sample = np.array(trial_samples)[np.where(predictions == attack_class)[0][0]]
                                break
                            else:
                                delta *= 0.9
                            # if d_step >= 5000:
                            #     break
                        # print("\tEpsilon step...")
                        # print("\td step finish")
                        # print("predict ", model.predict(moveaxis_change_channel_order(adversarial_sample))[0][0])
                        e_step = 0
                        while True:
                            e_step += 1
                            # print("\t#{}".format(e_step))
                            trial_sample = adversarial_sample + forward_perturbation(epsilon * get_diff(adversarial_sample, target_sample), adversarial_sample, target_sample)
                            prediction = model.predict(moveaxis_change_channel_order(trial_sample))
                            n_calls += 1
                            if (prediction > confidence_limit)[0][0].astype("int32") == attack_class:
                                adversarial_sample = trial_sample
                                epsilon /= 0.5
                                break
                            elif e_step > 5000:
                                break
                            else:
                                epsilon *= 0.5
                        # print("\te step finish")
                        # print("predict ", model.predict(moveaxis_change_channel_order(adversarial_sample))[0][0])
                        n_steps += 1
                        diff = np.mean(get_diff(adversarial_sample, target_sample))
                        if n_steps >= step_limit:
                            adversarial_sample = np.clip(adversarial_sample, 0, 1)
                            print("attack_class ", attack_class)
                            confidence = model.predict(moveaxis_change_channel_order(adversarial_sample))[0][0]
                            if (confidence > confidence_limit).astype("int32") == attack_class:
                                # draw(np.copy(moveaxis_change_channel_order(adversarial_sample)), model, os.path.join(path_save, dname,'n_steps_' + str(n_steps) + '_real_' + random_image.replace(".png", "") + '_attack_' + img_name))
                                plt.imsave(os.path.join(path_save, dname, f'n_steps_{str(n_steps)}_real_{random_image.replace(".png", "")}_attack_{img_num}_confidence_{confidence}.png'), adversarial_sample[0])
                                break
                        print("Mean Squared Error: {}".format(diff))
                        # print("Calls: {}".format(n_calls))
                        # print("Attack Class: {}".format(attack_class))
                        # print("Target Class: {}".format(target_class))
                        # print("Adversarial Class: {}".format(np.argmax(prediction)))
                        # print("Adversarial Class: {}".format((prediction > 0.5)[0][0].astype("int32")))
                    if (prediction > 0.5)[0][0].astype("int32") == attack_class:
                        attack_success += 1
                    else:
                        attack_fail += 1
                    print("attack_success: ", attack_success)
                    print("attack_fail: ", attack_fail)
                    print("strange_attack_fail: ", strange_attack_fail)

if __name__ == "__main__":
    boundary_attack()
