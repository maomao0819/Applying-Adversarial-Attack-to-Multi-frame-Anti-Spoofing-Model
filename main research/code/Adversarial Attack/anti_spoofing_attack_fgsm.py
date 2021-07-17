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

def origin_succes_rate(data_dir, img_width, img_height, model):
    total_attack = 0
    success_cnt = 0
    for idx, d in enumerate(os.listdir(data_dir)):
        filename = data_dir + d
        img = read_preprocess_image(filename, img_width, img_height)
        # print("aaa", img)
        outLabel = (model.predict(img) > 0.5).astype("int32")
        total_attack += 1
        if outLabel == 0:
            success_cnt += 1
    if success_cnt:
        if total_attack > 0:
            success_rate = success_cnt / total_attack
            print('accuracy: {:.2f}%'.format(100 * success_rate))
        else:
            print('accuracy: {:.2f}%'.format(100))
    else:
        print('accuracy: {:.2f}%'.format(0))

def create_adversarial_pattern(input_image, input_label, model):
    loss_object = tf.keras.losses.BinaryCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        #print("pre ", prediction)
        if np.array(prediction) <= 1e-9:
            prediction += 1e-9
        while np.array(prediction) <= 1e-7:
            prediction *= 10
        #print("pre", prediction)
        #print("input label", input_label)
        loss = loss_object(input_label, prediction)
        #print("loss", loss)
    #print("los ", loss)
    #print("input_image", input_image)
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    #print("gra ", gradient)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    # print("signed_grad ", np.array(signed_grad))
    return signed_grad

def perturbations(image, index, model):
    image = tf.convert_to_tensor(image)
    image_probs = model.predict(image)
    # label = tf.one_hot(index, image_probs.shape[-1])
    label = tf.one_hot(index, 2)
    # label = tf.reshape(label, (1, image_probs.shape[-1]))
    label = tf.reshape(label, (1, 2))
    return create_adversarial_pattern(image, label, model)

# plt.imshow(perturbations(image, tmp_label)[0] * 0.5 + 0.5); # To change [-1, 1] to [0,1]

def perturbations_t(perturbations):
    return tf.transpose(perturbations, perm = [0, 2, 3, 1])

def fgsm(image, eps, model):
    return image - eps * perturbations(image, 0, model)

def clip_0_1(origin_image, adv_x, eps):
    # origin_image = np.array((origin_image * 0.5 + 0.5) * 255) #-1 ~ 1 to 0 ~ 255
    origin_image = np.array(origin_image)
    # adv_x = np.array((adv_x * 0.5 + 0.5) * 255)
    adv_x = np.array(adv_x)
    # eps *= 255
    for x in range(adv_x.shape[1]):
        for y in range(adv_x.shape[2]):
            for z in range(adv_x.shape[3]):
                # adv_x[0, x, y, z] = min(255, origin_image[0, x, y, z] + eps, max(0, origin_image[0, x, y, z] - eps, adv_x[0, x, y, z]))
                adv_x[0, x, y, z] = min(1, origin_image[0, x, y, z] + eps, max(0, origin_image[0, x, y, z] - eps, adv_x[0, x, y, z]))
                # return tf.convert_to_tensor((adv_x / 255 - 0.5) * 2)
    return tf.convert_to_tensor(adv_x)

def ifgsm_0_1(origin_image, eps, model):
    # iteration_time_N = min(eps + 4, 1.25 * eps * 5)
    if eps > 1:
        iteration_time_N = min(eps + 4, 1.25 * eps)
    elif eps > 0:
        iteration_time_N = min(eps * 255 + 4, 1.25 * eps * 255)
    else:
        iteration_time_N = 0
    int_iteration_time_N = np.round(iteration_time_N, decimals=0).astype(np.int32)
    if int_iteration_time_N:
        # weight_alpha = np.round(eps / iteration_time_N if (eps / iteration_time_N) > 1 else 1)
        weight_alpha = eps / iteration_time_N
    adv_x = origin_image
    # print(eps, " ", int_iteration_time_N)
    for i in range(int_iteration_time_N):
        adv_x = clip_0_1(origin_image, adv_x - weight_alpha * perturbations(adv_x, 0, model), eps)
    return adv_x

def label_confidence_image(image, model):
    image_copy = image
    image_probs = model.predict(image_copy)
    outLabel = (model.predict(image_copy) > 0.5).astype("int32")
    # pred_class = np.argmax(image_probs, axis=-1)
    label = outLabel
    # print("image_probs", image_probs)
    confidence = np.amax(image_probs)
    # print("confidence", confidence)
    img = np.array(image_copy)
    img = np.moveaxis(img, 1, -1)
    img = img[0] * 0.5 + 0.5
    return label, confidence, img

def display_images(image, description, model):
    image_copy = image
    image_probs = model.predict(image_copy)
    outLabel = (model.predict(image_copy) > 0.5).astype("int32")
    # pred_class = np.argmax(image_probs, axis=-1)
    label = outLabel
    print("image_probs", image_probs)
    confidence = np.amax(image_probs)
    print("confidence", confidence)
    image_copy = np.array(image_copy)
    # plt.figure()
    image_copy = np.moveaxis(image_copy, 1, -1)
    # plt.imshow(image_copy[0]*0.5+0.5)
    # plt.title('{} \n {} : {:.2f}% Confidence'.format(description, label, confidence*100))
    # plt.show()

def CheckDirectory(path_save):
    if not os.path.exists(path_save):
        pathlib.Path(path_save).mkdir()

def anti_spoofying_attack(name_list, data_dir, dirs, img_width, img_height, model, epsilons):
    not_attack = 0
    attack_success = 0
    attack_fail = 0
    idx = 0
    # path_save = "adversarial_example_fgsm_training_data/"
    # path_save = "adversarial_example_fgsm_testing_data/"
    # path_save = "adversarial_example_ifgsm_testing_data/"
    path_save = "adversarial_example_fgsm/"
    CheckDirectory(path_save)
    CheckDirectory(path_save + dirs)
    descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')for eps in epsilons]

    for img_name in name_list:
        idx += 1
        image_dir = data_dir + dirs + '/' + img_name
        print(image_dir)
        test_img = read_preprocess_image(image_dir, img_width, img_height)
        # outLabel = (model.predict(test_img) > 0.5).astype("int32")
        label, confidence, adv_x = label_confidence_image(test_img, model)
        if label == 1:
            not_attack += 1
            # cv2.imwrite(path_save + "adversarial_example_no_attack_confidence_" + str(confidence) + ".png", adv_x)
            plt.imsave(path_save + "adversarial_example_no_attack_confidence_" + str(confidence) + ".png", adv_x)
            print(str(idx))
            print("not_attack: ", not_attack)
            print("attack_success: ", attack_success)
            print("attack_fail: ", attack_fail)
            continue
        # perturbations = perturbations(test_img, 0, model)
        # perturbations_t = perturbations_t(perturbations)
        # plt.title("perturbations")
        # plt.imshow((perturbations_t[0] * 0.5 + 0.5)) # To change [-1, 1] to [0,1]
        label = 0
        confidence_list = []
        eps_list = []
        adv_list = []
        for i, eps in enumerate(epsilons):
            # adv_x = ifgsm_0_1(test_img, eps, model)
            adv_x = fgsm(test_img, eps, model)
            adv_x = tf.clip_by_value(adv_x, -1, 1)
            # label = display_images(adv_x, descriptions[i], model)
            label, confidence, adv_x = label_confidence_image(adv_x, model)
            #print("eps ", eps, "confidence ", confidence)
            if label == 1:
                confidence_list.append(confidence)
                eps_list.append(eps)
                adv_list.append(adv_x)
                # attack_success += 1
        if len(confidence_list) == 0:
            attack_fail += 1
        else:
            attack_success += 1
            max_confidence_id = confidence_list.index(max(confidence_list))
            # cv2.imwrite(path_save + "adversarial_example_eps_" + str(eps_list[max_confidence_id]) + "_confidence_" + str(confidence_list[max_confidence_id]) + ".png", adv_list[max_confidence_id])
            plt.imsave(path_save + dirs + "/adversarial_example_" + img_name[:-4] + "_eps_" + str(eps_list[max_confidence_id]) + "_confidence_" + str(confidence_list[max_confidence_id]) + ".png", adv_list[max_confidence_id])
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
    f = open(path_save + "attack success rate.txt", "a")
    f.write(dirs + " :  "  + str(attack_success_rate) + "\n")
    f.close()

if __name__ == "__main__":
    # plt.axis('off')
    # plt.show()
    # plt.figure(figsize=(8,8)); plt.imshow(testing.astype(np.int)); plt.axis('off'); plt.show()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
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
    #if os.path.exists("adversarial_example_fgsm/attack success rate.txt"):
    #    os.remove("adversarial_example_fgsm/attack success rate.txt")
    for dirs in os.listdir(data_dir):
        
        print("dir ", dirs)

        origin_succes_rate(data_dir + dirs + '/', img_width, img_height, model)
        #print("aaa")
        #epsilons = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.35, 0.4, 0.5, 0.7, 0.9, 1.0]
        epsilons = [0.075, 0.1]
        img_name_list = name_list_gen(data_dir + dirs + '/')
        # print("img_name_list", img_name_list)
        anti_spoofying_attack(img_name_list, data_dir, dirs, img_width, img_height, model, epsilons)
