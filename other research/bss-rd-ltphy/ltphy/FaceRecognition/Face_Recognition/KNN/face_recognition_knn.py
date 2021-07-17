"""
This is an example of using the k-nearest-neighbors (KNN) algorithm for face recognition.

When should I use this example?
This example is useful when you wish to recognize a large set of known people,
and make a prediction for an unknown person in a feasible computation time.

Algorithm Description:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in an unknown image by finding the k most similar faces (images with closet face-features under eucledian distance)
in its training set, and performing a majority vote (possibly weighted) on their label.

For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden
and two images of Obama, The result would be 'Obama'.

* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.

Usage:

1. Prepare a set of images of the known people you want to recognize. Organize the images in a single directory
   with a sub-directory for each known person.

2. Then, call the 'train' function with the appropriate parameters. Make sure to pass in the 'model_save_path' if you
   want to save the model to disk so you can re-use the model without having to re-train it.

3. Call 'predict' and pass in your trained model to recognize the people in an unknown image.

NOTE: This example requires scikit-learn to be installed! You can install it with pip:

$ pip3 install scikit-learn

"""

import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw, ImageFont
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import time

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

number_of_no_face = 0

# def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    # X = []
    # y = []
    #h = 0.02
    # Loop through each person in the training set
    # for class_dir in os.listdir(train_dir):
    #     if not os.path.isdir(os.path.join(train_dir, class_dir)):
    #         continue

        # Loop through each training image for the current person
        # for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
        #     image = face_recognition.load_image_file(img_path)
        #     face_bounding_boxes = face_recognition.face_locations(image)

            # if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
            #     if verbose:
            #         print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            # else:
                # Add face encoding for current image to the training set
                # X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                # y.append(class_dir)
                # print(np.shape(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]))
                # print(image)
                # print(img_path)
                # print(X[len(X)-1])

    # Determine how many neighbors to use for weighting in the KNN classifier
    # if n_neighbors is None:
    #     n_neighbors = int(round(math.sqrt(len(X))))
    #     print(n_neighbors)
    #     if verbose:
    #         print("Chose n_neighbors automatically:", n_neighbors)
    # Create and train the KNN classifier
    # knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    #fit data into knn model
    # knn_clf.fit(X, y)

       # Save the trained KNN classifier
    # if model_save_path is not None:
    #     with open(model_save_path, 'wb') as f:
    #         pickle.dump(knn_clf, f)
            
    # return knn_clf


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    global X_face_locations
    X_face_locations = face_recognition.face_locations(X_img,  number_of_times_to_upsample=2, model = "cnn")
    # If no faces are found in the image, return an empty result.
    global number_of_no_face
    global length
    length = len(X_face_locations)

    if len(X_face_locations) == 0:
        number_of_no_face = number_of_no_face + 1
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=2) # return the closest distances of a current point to other neighbors
   	#print(closest_distances)
    #for i in range(len(X_face_locations)):
    	#print(closest_distances[0][i][0])
    are_matches = [min(closest_distances[0][i][0],closest_distances[0][i][1]) <= distance_threshold for i in range(len(X_face_locations))]
    #print(are_matches)
    # Predict classes and remove classifications that aren't within the threshold
    predictions = []
    global pred
    for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches):
        print("Encode:",pred, loc, rec)
        
        if rec:
            predictions.append((pred,loc))
        else:
            predictions.append(("unknown",loc))
  
    #return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
    return predictions

# def show_prediction_labels_on_image(count, img_path, predictions):
#     """  
#     Shows the face recognition results visually.

#     :param img_path: path to image to be recognized
#     :param predictions: results of the predict function
#     :return:
#     """
#     pil_image = Image.open(img_path).convert("RGB")
#     draw = ImageDraw.Draw(pil_image)
#     global name
#     for name, (top, right, bottom, left) in predictions:
#         # Draw a box around the face using the Pillow module
#         # bottom = bottom  + 30
#         draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
#         # font = ImageFont.truetype("arial.ttf",20)
#         # There's a bug in Pillow where it blows up with non-UTF-8 text
#         # when using the default bitmap font
#        # name = name.encode("UTF-8") return byte
#         #print(type(name))
#        #	font = ImageFont.truetype("C:/Users/Phy/Desktop/OpenFace/KNN/arial.ttf", 30)
#         # Draw a label with a name below the face
#         w,h = font.getsize(name) #keep as it is string
#         print("my name:",name)
#         #text_width, text_height = draw.textsize(name)
#         draw.rectangle(((left, bottom - h - 10), (max(right,left+w+8) , bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
#         draw.text((left + 6, bottom - h-5), name, font= font)#,font = font)

#     # Remove the drawing library from memory as per the Pillow docs
#     del draw

#     # Display the resulting image
#     path = r'C:\Users\imf\Downloads\bss-rd-student-ltphy\bss-rd-student-ltphy\15062018_ltphy\FaceRecognition\Face_Recognition\KNN\outputs'
#     pil_image.show()
#     pil_image.save(os.path.join(path, "result" + str(count)+".png"))

if __name__ == "__main__":
    #global number_of_no_face
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    # print("Training KNN classifier...")
    # classifier = train("knn_examples/train/face recognition train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    #time.sleep(30)
    # print("Training complete!")

    # STEP 2: Using the trained classifier, make predictions for unknown images
    count = 0
    num = 0
    number_of_image = 0
    correct = 0
    z = []
    # for _class_ in os.listdir("knn_examples/test/adversarial_example_fgsm"):
    #     z.append(_class_)

    # for num in range(20):
    #     for image_file in os.listdir("knn_examples/test/adversarial_example_fgsm/"+ z[num]):
    #         number_of_image = number_of_image + 1
    #         print("number_of_image:", number_of_image)
    #         full_file_path = os.path.join("knn_examples/test/adversarial_example_fgsm/"+ z[num], image_file)

    #         # print("Looking for faces in {}".format(image_file))

    #         # Find all people in the image using a trained classifier model
    #         # Note: You can pass in either a classifier file name or a classifier model instance
    #         predictions = predict(full_file_path, model_path="trained_knn_model.clf")
    #         pil_image = Image.open(full_file_path).convert("RGB")
    #         pil_image = np.array(pil_image)            
    #         for name, (top, right, bottom, left) in predictions:
    #             pil_image = pil_image[left:right][top:bottom]
    #             pil_image = Image.fromarray(pil_image)
    #             pil_image.save(os.path.join("outputs", image_file))

    #         # Print results on the console
    #         # for name, (top, right, bottom, left) in predictions:
    #         #     print("- Found {} at ({}, {})".format(name, left, top))


    #         # Display results overlaid on an image
    #         # show_prediction_labels_on_image(count, full_file_path, predictions)
    #         if len(predictions):
    #             if(predictions[0][0] == z[num] and length != 0):
    #                 correct = correct + 1
    #             count = count + 1
    #             print("no face:", number_of_no_face)

    for image_file in os.listdir("knn_examples/test/homemade_test_rebroadcast"):
        number_of_image = number_of_image + 1
        print("number_of_image:", number_of_image)
        full_file_path = os.path.join("knn_examples/test/homemade_test_rebroadcast", image_file)

        predictions = predict(full_file_path, model_path="trained_knn_model.clf")
        pil_image = Image.open(full_file_path).convert("RGB")           
        for name, (top, right, bottom, left) in predictions:
            pil_image = pil_image.crop((left,top,right,bottom))
            pil_image.save(os.path.join("outputs_PSNR", image_file))

        if len(predictions):
            print("predictions:", predictions[0][0])
            if(predictions[0][0] == pred and length != 0):
                correct = correct + 1
            count = count + 1
        print("no face:", number_of_no_face)

    
    # accuracy = (correct/number_of_image)*100
    # error_predict = number_of_image - correct - number_of_no_face

    # print("correct:", correct)
    # print("error predict:", error_predict)
    # print("accuracy:", accuracy)
