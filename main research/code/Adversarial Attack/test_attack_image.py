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
	# plt.axis('off')
	# plt.show()
	# plt.figure(figsize=(8,8)); plt.imshow(testing.astype(np.int)); plt.axis('off'); plt.show()

	model_path = 'anti_spoofing_model.h5'

	if model_path:
		model = load_model(model_path)
	else:
		print('Could not load model!')

	img_path = 'adversarial_example_ifgsm_training_data'
	attack_success = 0
	attack_fail = 0
	idx = 0
	camera_number = 1
	cap = cv2.VideoCapture(camera_number + cv2.CAP_DSHOW)
	img_width, img_height = (96, 96)
	for image in os.listdir(img_path):
		idx += 1
		img = cv2.imread(img_path + '/' + image)
		img = cv2.resize(img, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
		cv2.imshow('img', img)
		ret, frame = cap.read()
		if ret == False:
			continue
		else:
			cv2.imshow('frame', frame)
			k = cv2.waitKey(3)
			test_img = preprocess_image(frame, img_width, img_height)
			label, confidence, adv_x = label_confidence_image(test_img, model)
			if label == 1:
				attack_success += 1
			else:
				attack_fail += 1
			print("path", img_path + '/' + image)
			print(str(idx))
			print("attack_success: ", attack_success)
			print("attack_fail: ", attack_fail)
	cap.release()
	cv2.destroyAllWindows()
	print("attack_success_rate: ", attack_success / (attack_success + attack_fail))