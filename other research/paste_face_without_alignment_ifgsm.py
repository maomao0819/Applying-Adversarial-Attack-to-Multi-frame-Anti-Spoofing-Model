import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from PIL import Image
import os
import pathlib

def CheckDirectory(path_save):
    if not os.path.exists(path_save):
        pathlib.Path(path_save).mkdir()

def detectFace(img, face_detector):
	faces = face_detector.detectMultiScale(img, 1.3, 5)
	#print("found faces: ", len(faces))
	face_x = face_y = face_w = face_h = 1
	if len(faces) > 0:
		face = faces[0]
		face_x, face_y, face_w, face_h = face
	return face_x, face_y, face_w, face_h

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

def paste_face_without_alignment():
	face_detector, eye_detector, nose_detector = cv2_detector()
	path_read_origin_image = os.path.join('anti-spoofing test origin image', 'attack_choose')
	path_read_face_image = "adversarial_example_ifgsm"
	path_save = "image_paste_face_without_alignment_ifgsm"
	CheckDirectory(path_save)
	extension = '.png'
	for dname in os.listdir(path_read_face_image):
		print("dname: ", dname)
		CheckDirectory(os.path.join(path_save, dname))
		if os.path.isdir(os.path.join(path_read_face_image, dname)):
			for image_path in os.listdir(os.path.join(path_read_face_image, dname)):
				face_image = cv2.imread(os.path.join(path_read_face_image, dname, image_path))
				image_path = f"{str(image_path.split('_')[2])}{extension}"
				origin_image = cv2.imread(os.path.join(path_read_origin_image, dname, image_path))
				face_x, face_y, face_w, face_h = detectFace(origin_image, face_detector)
				origin_image[int(face_y):int(face_y + face_h), int(face_x):int(face_x + face_w)] = cv2.resize(face_image, (face_w, face_h), interpolation=cv2.INTER_NEAREST)
				cv2.imwrite(os.path.join(path_save, dname, image_path), origin_image)

if __name__ == "__main__":
	paste_face_without_alignment()
