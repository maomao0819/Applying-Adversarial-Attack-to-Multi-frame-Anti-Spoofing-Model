import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from PIL import Image
import os
import pathlib
import tensorflow as tf
from tensorflow.keras.models import load_model

def CheckDirectory(path_save):
	if not os.path.exists(path_save):
		pathlib.Path(path_save).mkdir()

def detectFace(img, face_detector):
	faces = face_detector.detectMultiScale(img, 1.3, 5)
	if len(faces) > 0:
		face = faces[0]
		face_x, face_y, face_w, face_h = face
		img = img[int(face_y):int(face_y+face_h), int(face_x):int(face_x+face_w)]
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		return img, img_gray
	else:
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		return img, img_gray
		raise ValueError("No face found in the passed image ")

def cv2_detector():
	#opencv path
	opencv_home = cv2.__file__

	folders = opencv_home.split(os.path.sep)[0:-1]
	path = os.path.join(*folders)
	face_detector_path = f"/{os.path.join(path, 'data', 'haarcascade_frontalface_default.xml')}"

	# path = folders[0]
	# for folder in folders[1:]:
	# 	path = path + "/" + folder
	# face_detector_path = path + "/data/haarcascade_frontalface_default.xml"

	eye_detector_path = path + "/data/haarcascade_eye.xml"
	nose_detector_path = path + "/data/haarcascade_mcs_nose.xml"

	if os.path.isfile(face_detector_path) != True:
		raise ValueError("Confirm that opencv is installed on your environment! Expected path ", face_detector_path," violated.")

	face_detector = cv2.CascadeClassifier(face_detector_path)
	eye_detector = cv2.CascadeClassifier(eye_detector_path)
	nose_detector = cv2.CascadeClassifier(nose_detector_path)
	return face_detector, eye_detector, nose_detector

def crop_face_and_model_test():
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

	face_detector, eye_detector, nose_detector = cv2_detector()
	attack = "boundary_attack"
	path_read = f"image_paste_face_without_alignment_{attack}"
	extension = '.png'
	for dname in os.listdir(path_read):
		print("dname: ", dname)
		if os.path.isdir(os.path.join(path_read, dname)):
			attack_success = 0
			attack_fail = 0
			for image_path in os.listdir(os.path.join(path_read, dname)):
				origin_image = cv2.imread(os.path.join(path_read, dname, image_path))
				face_image = detectFace(origin_image, face_detector)
				label = (model.predict(face_image) > 0.5).astype("int32")
				if label == 1:
					attack_success += 1
				else:
					attack_fail += 1
			with open(os.path.join(path_read, "attack success rate.txt"), 'a') as fp:
				fp.write(f"{dname} :  {str(attack_success / (attack_success + attack_fail))}\n")

if __name__ == "__main__":
	crop_face_and_model_test()