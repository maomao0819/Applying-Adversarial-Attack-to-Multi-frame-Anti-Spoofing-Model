import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import imgaug as ia
import imgaug.augmenters as iaa
tf.keras.backend.set_image_data_format('channels_first')
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def to_channel_last(image):
    shape = np.shape(image)
    if shape[-1] != 3:
        if shape[0] == 3:
            return np.moveaxis(image, 0, -1)
        if shape[1] == 3:
            return np.moveaxis(image, 1, -1)
    return image 

# seq = iaa.Sequential([
#     # Apply affine transformations to each image.
#     # Scale/zoom them, translate/move them, rotate them and shear them.
#     iaa.Affine(
#         rotate=(-5, 5),
#         shear=(-5, 5),
#         scale={"x": (0.85, 1.15), "y": (0.85, 1.15)},
#         translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}
#     ),

#     iaa.PerspectiveTransform(scale=(0, 0.025)),

#     # Make some images brighter and some darker.
#     # In 20% of all cases, we sample the multiplier once per channel,
#     # which can end up changing the color of the images.
#     # iaa.Multiply((0.8, 1.2), per_channel=0.2),
#     iaa.Multiply((0.85, 1.15)),

#     # Strengthen or weaken the contrast in each image.
#     iaa.LinearContrast((0.9, 1.1)),

#     # Small gaussian blur with random sigma between 0 and 0.5.
#     # But we only blur about 50% of all images.
#     # iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
#     iaa.GaussianBlur(sigma=(0, 1)),
        
#     iaa.AddToHueAndSaturation((-15, 15), per_channel=True)

#     # iaa.Fliplr(0.5), # horizontal flips
#     # iaa.Crop(percent=(0, 0.1)), # random crops
        
#     # Add gaussian noise.
#     # For 50% of all images, we sample the noise once per pixel.
#     # For the other 50% of all images, we sample the noise per pixel AND
#     # channel. This can change the color (not only brightness) of the
#     # pixels.
#     # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            
# ], random_order=True) # apply augmenters in random order

# def image_augment(images):
#     images = to_channel_last(images)
#     images_augment = seq(images=images.astype(np.uint8))
#     if np.shape(images_augment)[-1] == 3:
#         images_augment = np.moveaxis(images_augment, -1, 1)
#     return images_augment

def read_preprocess_image(imgPath, img_width, img_height):
    img = load_img(imgPath, target_size=(img_width, img_height))
    imgArray = img_to_array(img)
    imgArray = imgArray.reshape(1, 3, img_width, img_height)
    # img_aug = image_augment(imgArray)
    imgArray = imgArray / float(255)
    return imgArray

def succes_rate(data_dir, img_width, img_height, model):
    total_attack = 0
    success_cnt = 0
    images = []
    for image_name in os.listdir(data_dir):
        extension = os.path.splitext(image_name)[1]
        if extension != '.png':
            continue
        if image_name.split('_')[-1] == 'fix.png':
            continue
        filename = os.path.join(data_dir, image_name)
        img = read_preprocess_image(filename, img_width, img_height)
        images.append(img[0])
        print("confidence ", model.predict(img))
        outLabel = (model.predict(img) > 0.5).astype("int32")
        total_attack += 1
        if outLabel == 1:
            success_cnt += 1
        # elif outLabel == 0:
        #     os.remove(filename)
    if success_cnt:
        if total_attack > 0:
            success_rate = success_cnt / total_attack
            print(f'accuracy: {100 * success_rate:.2f}%')
        else:
            print(f'accuracy: {100:.2f}%')
    else:
        print(f'accuracy: {0:.2f}%')
    return images

if __name__ == "__main__":
    # model_path = 'REPLAY_homemade.h5'
    model_path = os.path.join('anti_spoofing_optimize', 'REPLAY_homemade_776_half_epoch2.h5')
    # image_path = os.path.join('cropped_image_homemade', 'outputs_larger')
    image_path = os.path.join('anti_spoofing_optimize', 'adversarial_example_boundary_attack_for_testing_data_test_100_124')
    if model_path:
        model = load_model(model_path)
    else:
        print('Could not load model!')
    img_width, img_height = (96, 96)
    for image_type in os.listdir(image_path):
        print(image_type)

        images = succes_rate(os.path.join(image_path, image_type), img_width, img_height, model)
        images = np.array(images)
        num = np.shape(images)[0]
        # labels = np.array([1] * num)
        # loss, acc = model.evaluate(images, labels)
        # print("acc", acc)

