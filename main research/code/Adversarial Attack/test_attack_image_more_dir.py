import numpy as np
import os
import pathlib
import cv2
# Load Keras dependencies:
import tensorflow
from tensorflow.keras.preprocessing import image
tensorflow.keras.backend.set_image_data_format('channels_first')

import glob

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def preprocess_image(img, img_width, img_height):
	img = np.resize(img, (3, img_width, img_height))
	imgArray = img_to_array(img)
	imgArray = imgArray.reshape(1, 3, img_width, img_height)
	imgArray = imgArray / float(255)
	return imgArray

def CheckDirectory(path_save):
	if not os.path.exists(path_save):
		pathlib.Path(path_save).mkdir()

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

if __name__ == "__main__":

	model_path = 'REPLAY-HalfFPS_val96.h5'

	if model_path:
		model = load_model(model_path)
	else:
		print('Could not load model!')

	camera_number = 1
	cap = cv2.VideoCapture(camera_number + cv2.CAP_DSHOW)
	img_width, img_height = (96, 96)

	path_read = 'adversarial_example_fgsm_range'
	path_save = os.path.join('adversarial_example_fgsm_range', 'adversarial_example_fgsm_range_attack_success_rate.txt')
	if os.path.exists(path_save):
		os.remove(path_save)
	for dirs in os.listdir(path_read):
		if os.path.isdir(os.path.join(path_read, dirs)):
			attack_success = 0
			attack_fail = 0
			attack_success_reshot = 0
			attack_fail_reshot = 0
			for images in os.listdir(os.path.join(path_read, dirs)):
				img = cv2.imread(os.path.join(path_read, dirs, images))
				test_img = preprocess_image(img, img_width, img_height)
				label, confidence, adv_x = label_confidence_image(test_img, model)
				if label == 1:
					attack_success += 1
				else:
					attack_fail += 1
				img = cv2.resize(img, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
				cv2.imshow('img', img)
				ret, frame = cap.read()
				if ret == False:
					continue
				else:
					cv2.imshow('frame', frame)
					k = cv2.waitKey(3)
					test_img_reshot = preprocess_image(frame, img_width, img_height)
					label_reshot, confidence_reshot, adv_x_reshot = label_confidence_image(test_img_reshot, model)
					if label_reshot == 1:
						attack_success_reshot += 1
					else:
						attack_fail_reshot += 1
					print("attack_success_reshot: ", attack_success_reshot)
					print("attack_fail_reshot: ", attack_fail_reshot)
			attack_success_rate_reshot = attack_success_reshot / (attack_success_reshot + attack_fail_reshot)
			f = open(path_save, "a")
			f.write(f"{dirs} :  {str(attack_success_rate)}\n")
			f.write(f"{dirs} :  {str(attack_success_rate_reshot)}\n")
			f.close()
	cap.release()
	cv2.destroyAllWindows()
