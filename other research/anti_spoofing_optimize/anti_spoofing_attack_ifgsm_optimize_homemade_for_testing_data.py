import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import pathlib
import math
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

# Load imgaug
import imgaug as ia
import imgaug.augmenters as iaa
from tensorflow.keras import backend as K
import gc
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def to_channel_last(image):
    shape = np.shape(image)
    if shape[-1] != 3:
        if shape[0] == 3:
            return np.moveaxis(image, 0, -1)
        if shape[1] == 3:
            return np.moveaxis(image, 1, -1)
    return image 

seq = iaa.Sequential([
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        rotate=(-5, 5),
        shear=(-5, 5),
        scale={"x": (0.85, 1.15), "y": (0.85, 1.15)},
        translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}
    ),

    iaa.PerspectiveTransform(scale=(0, 0.025)),

    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    # iaa.Multiply((0.8, 1.2), per_channel=0.2),
    iaa.Multiply((0.85, 1.15)),

    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.9, 1.1)),

    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    # iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
    iaa.GaussianBlur(sigma=(0, 1)),
        
    iaa.AddToHueAndSaturation((-15, 15), per_channel=True)

    # iaa.Fliplr(0.5), # horizontal flips
    # iaa.Crop(percent=(0, 0.1)), # random crops
        
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            
], random_order=True) # apply augmenters in random order

def image_augment(images):
    images = to_channel_last(images)
    images_augment = seq(images=images.astype(np.uint8))
    if np.shape(images_augment)[-1] == 3:
        images_augment = np.moveaxis(images_augment, -1, 1)
    return images_augment

def read_preprocess_image(imgPath, img_width, img_height):
    img = load_img(imgPath, target_size=(img_width, img_height))
    imgArray = img_to_array(img)
    imgArray = imgArray.reshape(1, 3, img_width, img_height)
    # img_aug = image_augment(imgArray)
    imgArray = imgArray / float(255)
    # imgArray_list = []
    # for _ in range(2000):
    #     img_aug = image_augment(imgArray)
    #     img_aug = img_aug / float(255)
    #     imgArray_list.append(img_aug)
    # imgArray = imgArray_list[0]
    return imgArray

#attack label: 0
#real label: 1

def origin_succes_rate(data_dir, img_width, img_height, model):
    total_attack = 0
    success_cnt = 0
    for image_name in os.listdir(data_dir):
        print("image_name ", image_name)
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

def prediction_fix(prediction, is_prediction_fix):
    # if np.array(prediction) <= 1e-9:
    #     prediction += 1e-9
    while np.array(prediction) <= 1e-7:
        prediction *= 10
        is_prediction_fix = True
    return prediction, is_prediction_fix

def create_adversarial_pattern(input_image, input_label, model):
    loss_object = tf.keras.losses.BinaryCrossentropy()
    is_prediction_fix = False
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        # print("pred ", np.array(prediction[0]))
        prediction, is_prediction_fix = prediction_fix(prediction, is_prediction_fix)
        loss = loss_object(input_label, prediction)
        for _ in range(799):
            K.clear_session()
            gc.collect()
            img_aug = image_augment(input_image * float(255)) / float(255)
            prediction = model(img_aug)
            prediction, _ = prediction_fix(prediction, is_prediction_fix)
            # print('prediction ', prediction)
            loss += loss_object(input_label, prediction)
        # print("loss ", loss)
        loss /= 800
    gradient = tape.gradient(loss, input_image)
    # print("grad ", gradient)
    signed_grad = tf.sign(gradient)
    return signed_grad, is_prediction_fix

def perturbations(image, index, model):
    image = tf.convert_to_tensor(image)
    # image_probs = model.predict(image)
    # label = tf.one_hot(index, 2)
    # label = tf.reshape(label, (1, 2))
    label = tf.convert_to_tensor(index)
    return create_adversarial_pattern(image, label, model)

def perturbations_t(perturbations):
    return tf.transpose(perturbations, perm = [0, 2, 3, 1])

def fgsm(image, eps, model):
    perturbation, is_prediction_fix = perturbations(image, 0, model)
    return image + eps * perturbation, is_prediction_fix

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
    # int_iteration_time_N = 5
    if int_iteration_time_N:
        weight_alpha = eps / int_iteration_time_N
    adv_x = origin_image
    origin_image_np = np.array(origin_image)
    is_prediction_fix = False
    for _ in range(int_iteration_time_N):
        perturbation, is_prediction_fix_tmp = perturbations(adv_x, 0, model)
        if is_prediction_fix_tmp:
            is_prediction_fix = True
        adv_x = clip_0_1(origin_image_np, adv_x + weight_alpha * perturbation, eps)
    return adv_x, is_prediction_fix

def label_confidence_image(image, model):
    image_copy = image
    image_probs = model.predict(image_copy)
    label = (image_probs > 0.5).astype("int32")
    confidence = np.amax(image_probs)
    img = np.array(image_copy)
    img = np.moveaxis(img, 1, -1)
    img = img[0]
    return label, confidence, img

def CheckDirectory(path_save):
    if not os.path.exists(path_save):
        pathlib.Path(path_save).mkdir()

def calculate_psnr(img1, img2):
    if len(np.shape(img1)) == 4:
        img1 = np.squeeze(img1, axis=0)
    if len(np.shape(img2)) == 4:
        img2 = np.squeeze(img2, axis=0)
    if np.shape(img1)[0] == 3:
        img1 = np.moveaxis(img1, 0, -1)
    if np.shape(img2)[0] == 3:
        img2 = np.moveaxis(img2, 0, -1)
    mse = np.mean((img1 - img2) ** 2) * 255 * 255
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return psnr

def anti_spoofying_attack(name_list, data_dir, dirs, img_width, img_height, model, epsilons, path_save):
    idx = 0
    not_attack = 0
    attack_success = 0
    attack_fail = 0
    psnr_fail_first = 0
    CheckDirectory(path_save)
    CheckDirectory(os.path.join(path_save, dirs))
    for img_name in name_list:
        idx += 1
        image_dir = os.path.join(data_dir, dirs, img_name)
        print(image_dir)
        test_img = read_preprocess_image(image_dir, img_width, img_height)
        label, confidence, adv_x = label_confidence_image(test_img, model)
        if label == 1:
            not_attack += 1
            # plt.imsave(os.path.join(path_save, dirs, f"adversarial_example_no_attack_confidence_{str(confidence)}.png"), adv_x)
            print(str(idx))
            print("not_attack: ", not_attack)
            print("attack_success: ", attack_success)
            print("attack_fail: ", attack_fail)
            print("psnr_fail_first: ", psnr_fail_first)
            continue
        label = 0
        optimize_confidence = 0
        optimize_eps = 0
        optimize_adv = test_img
        for eps in epsilons:
            adv_x, is_prediction_fix = ifgsm_0_1(test_img, eps, model)
            # adv_x, is_prediction_fix = fgsm(test_img, eps, model)
            adv_x = tf.clip_by_value(adv_x, 0, 1)
            # label = display_images(adv_x, descriptions[i], model)
            label, confidence, adv_x = label_confidence_image(adv_x, model)
            print("eps ", eps, "confidence ", confidence)
            if confidence > optimize_confidence:
                (optimize_confidence, optimize_eps, optimize_adv) = (confidence, eps, adv_x)
        is_prediction_fix_str = ""
        if is_prediction_fix:
            is_prediction_fix_str += "_prediction_fix"
        if optimize_confidence < 0.5:
            attack_fail += 1
            plt.imsave(os.path.join(path_save, dirs, f"adversarial_example_{img_name[:-4]}_attack_fail{is_prediction_fix_str}.png"), optimize_adv)
        else:
            attack_success += 1
            is_psnr_pass_first = True
            psnr_str = ''
            print('PSNR ', calculate_psnr(test_img, optimize_adv))
            while calculate_psnr(test_img, optimize_adv) < 20 or optimize_confidence < 0.5:
                is_psnr_pass_first = False
                optimize_eps /= 0.9
                adv_x = fgsm(test_img, optimize_eps, model)
                adv_x = tf.clip_by_value(adv_x, 0, 1)
                label, optimize_confidence, optimize_adv = label_confidence_image(adv_x, model)
            if not is_psnr_pass_first:
                psnr_str = '_psnr_fail_first'
                psnr_fail_first += 1
            plt.imsave(os.path.join(path_save, dirs, f"adversarial_example_{img_name[:-4]}_eps_{str(optimize_eps)}_confidence_{str(optimize_confidence)}{psnr_str}{is_prediction_fix_str}.png"), optimize_adv)
        print(str(idx))
        print("not_attack: ", not_attack)
        print("attack_success: ", attack_success)
        print("attack_fail: ", attack_fail)
        print("psnr_fail_first: ", psnr_fail_first)
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
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs, ", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    # model_path = 'REPLAY-HalfFPS_val96.h5'
    model_path = os.path.join('REPLAY_homemade_776_half_epoch2.h5')
    if model_path:
        model = load_model(model_path)
    else:
        print('Could not load model!')
    opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
    model.compile(loss='binary_crossentropy',
    #model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
                  #metrics=['categorical_accuracy'])
    model.summary()

    img_width, img_height = (96, 96)

    name_list = []
    # data_dir = "anti-spoofing test attack image/"
    data_dir = 'homemade_attack_choose_cropped_clients_test_100_124'
    path_save = "adversarial_example_ifgsm_0.02_with_imgaug_with_PSNR_test_100_124"
    if os.path.exists(os.path.join(path_save, "attack success rate.txt")):
        os.remove(os.path.join(path_save, "attack success rate.txt"))
    for dirs in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, dirs)):
            print("dir ", dirs)
            if int(dirs.split('_')[2]) > 100:
                # origin_succes_rate(os.path.join(data_dir, dirs), img_width, img_height, model)
                # fgsm_epsilons = [0.03, 0.04]
                epsilons = [0.02]
                # ifgsm_epsilons = [i * 0.01 for i in range(10)]
                img_name_list = os.listdir(os.path.join(data_dir, dirs))
                anti_spoofying_attack(img_name_list, data_dir, dirs, img_width, img_height, model, epsilons, path_save)
