import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import pathlib
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def read_preprocess_image(imgPath, img_width, img_height):
    img = load_img(imgPath, target_size=(img_width, img_height))
    imgArray = img_to_array(img)
    imgArray = imgArray.reshape(1, 3, img_width, img_height)
    imgArray = imgArray / float(255)
    return imgArray

#attack label: 0
#real label: 1

def origin_succes_rate(data_dir, img_width, img_height, model):
    total_attack = 0
    success_cnt = 0
    for image_name in os.listdir(data_dir):
        filename = os.path.join(data_dir, image_name)
        img = read_preprocess_image(filename, img_width, img_height)
        outLabel = (model.predict(img) > 0.5).astype("int32")
        total_attack += 1
        if outLabel == 0:
            success_cnt += 1
    if success_cnt:
        if total_attack > 0:
            success_rate = success_cnt / total_attack
            print(f'accuracy: {100 * success_rate:.2f}%')
        else:
            print(f'accuracy: {100:.2f}%')
    else:
        print(f'accuracy: {0:.2f}%')

def create_adversarial_pattern(input_image, input_label, model):
    loss_object = tf.keras.losses.BinaryCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        if np.array(prediction) <= 1e-9:
            prediction += 1e-9
        while np.array(prediction) <= 1e-7:
            prediction *= 10
        loss = loss_object(input_label, prediction)
    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad

def perturbations(image, index, model):
    image = tf.convert_to_tensor(image)
    # image_probs = model.predict(image)
    label = tf.one_hot(index, 2)
    label = tf.reshape(label, (1, 2))
    return create_adversarial_pattern(image, label, model)

def perturbations_t(perturbations):
    return tf.transpose(perturbations, perm = [0, 2, 3, 1])

def fgsm(image, eps, model):
    return image - eps * perturbations(image, 0, model)

def clip_0_1(origin_image_np, adv_x, eps):
    adv_x = tf.clip_by_value(adv_x, 0, 1)
    adv_x = tf.clip_by_value(adv_x, origin_image_np - eps, origin_image_np + eps)
    return tf.convert_to_tensor(adv_x)
    
def ifgsm_0_1(origin_image, eps, model):
    if eps > 1:
        iteration_time_N = min(eps + 4, 1.25 * eps)
    elif eps > 0:
        iteration_time_N = min(eps * 255 + 4, 1.25 * eps * 255)
    else:
        iteration_time_N = 0
    int_iteration_time_N = np.round(iteration_time_N, decimals=0).astype(np.int32)
    if int_iteration_time_N:
        weight_alpha = eps / iteration_time_N
    adv_x = origin_image
    origin_image_np = np.array(origin_image)
    for i in range(int_iteration_time_N):
        adv_x = clip_0_1(origin_image_np, adv_x - weight_alpha * perturbations(adv_x, 0, model), eps)
    return adv_x

def label_confidence_image(image, model):
    image_copy = image
    image_probs = model.predict(image_copy)
    outLabel = (model.predict(image_copy) > 0.5).astype("int32")
    label = outLabel
    confidence = np.amax(image_probs)
    img = np.array(image_copy)
    img = np.moveaxis(img, 1, -1)
    img = img[0] * 0.5 + 0.5
    return label, confidence, img

def CheckDirectory(path_save):
    if not os.path.exists(path_save):
        pathlib.Path(path_save).mkdir()

def anti_spoofying_attack(name_list, data_dir, dirs, img_width, img_height, model, epsilons, path_save):
    not_attack = 0
    attack_success = 0
    attack_fail = 0
    idx = 0
    CheckDirectory(path_save)
    CheckDirectory(path_save + dirs)

    for img_name in name_list:
        idx += 1
        image_dir = os.path.join(data_dir, dirs, img_name)
        print(image_dir)
        test_img = read_preprocess_image(image_dir, img_width, img_height)
        label, confidence, adv_x = label_confidence_image(test_img, model)
        if label == 1:
            not_attack += 1
            plt.imsave(os.path.join(path_save, dirs, f"adversarial_example_no_attack_confidence_{str(confidence)}.png"), adv_x)
            print(str(idx))
            print("not_attack: ", not_attack)
            print("attack_success: ", attack_success)
            print("attack_fail: ", attack_fail)
            continue
        label = 0
        confidence_list = []
        eps_list = []
        adv_list = []
        for i, eps in enumerate(epsilons):
            adv_x = ifgsm_0_1(test_img, eps, model)
            # adv_x = fgsm(test_img, eps, model)
            adv_x = tf.clip_by_value(adv_x, -1, 1)
            # label = display_images(adv_x, descriptions[i], model)
            label, confidence, adv_x = label_confidence_image(adv_x, model)
            print("eps ", eps, "confidence ", confidence)
            if label == 1:
                confidence_list.append(confidence)
                eps_list.append(eps)
                adv_list.append(adv_x)
        if len(confidence_list) == 0:
            attack_fail += 1
        else:
            attack_success += 1
            max_confidence_id = confidence_list.index(max(confidence_list))
            plt.imsave(os.path.join(path_save, dirs, f"adversarial_example_{img_name[:-4]}_eps_{str(eps_list[max_confidence_id])}_confidence_{str(confidence_list[max_confidence_id])}.png"), adv_list[max_confidence_id])
        print(str(idx))
        print("not_attack: ", not_attack)
        print("attack_success: ", attack_success)
        print("attack_fail: ", attack_fail)
    attack_success_rate = 0
    if (attack_success + attack_fail + not_attack) > 0:
        attack_success_rate = attack_success / (attack_success + attack_fail + not_attack)
        print("attack_success_rate: ", attack_success_rate)
    elif attack_success:
        attack_success_rate = 100
        print("attack_success_rate: 100")
    else:
        print("attack_success_rate: 0")
    f = open(os.path.join(path_save, "attack success rate.txt"), "a")
    f.write(f"{dirs} :  {str(attack_success_rate)}\n")
    f.close()

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs, ", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    model_path = 'REPLAY-HalfFPS_val96.h5'

    if model_path:
        model = load_model(model_path)
    else:
        print('Could not load model!')

    model.summary()

    img_width, img_height = (96, 96)

    name_list = []
    data_dir = "anti-spoofing test attack image/"
    path_save = "adversarial_example_ifgsm_eps_0.05_or_0.06/"

    if os.path.exists(os.path.join(path_save, "attack success rate.txt")):
        os.remove(os.path.join(path_save, "attack success rate.txt"))
    for dirs in os.listdir(data_dir):
        print("dir ", dirs)

        origin_succes_rate(os.path.join(data_dir, dirs), img_width, img_height, model)
        # fgsm_epsilons = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        ifgsm_epsilons = [0.05, 0.06]
        img_name_list = os.listdir(data_dir + dirs + '/')
        anti_spoofying_attack(img_name_list, data_dir, dirs, img_width, img_height, model, ifgsm_epsilons, path_save)
