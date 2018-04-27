import dlib
import math
import numpy as np
import cv2
from utils import CLAHE, LANDMARKS_PREDICTOR_PATH


face_detector = dlib.get_frontal_face_detector()
landmarks_predictor = dlib.shape_predictor(LANDMARKS_PREDICTOR_PATH)


def get_landmarks_from_face(gray_image):
    if gray_image is None:
        return None
    face_detections = face_detector(gray_image, 1)

    landmarks_v = []

    # #Store X and Y coordinates in two lists
    for k, d in enumerate(face_detections):
        shape = landmarks_predictor(gray_image, d)
        x_list = []
        y_list = []
        for i in range(1, 68):
            x_list.append(float(shape.part(i).x))
            y_list.append(float(shape.part(i).y))

        # Find both coordinates of centre of gravity
        x_mean = np.mean(x_list)
        y_mean = np.mean(y_list)

        # To compensate for variation in location of face in the frame.
        x_central = [(x - x_mean) for x in x_list]
        y_central = [(y - y_mean) for y in y_list]

        for x, y, w, z in zip(x_central, y_central, x_list, y_list):
            # Co-ordinates are added relative to the Centre of gravity of face to accompany for
            landmarks_v.append(w)
            landmarks_v.append(z)

            # Euclidean distance between each point and the centre point (length of vector)
            np_mean_coor = np.asarray((y_mean, x_mean))
            np_coor = np.asarray((z, w))
            euclid_d = np.linalg.norm(np_coor-np_mean_coor)

            landmarks_v.append(euclid_d)
            landmarks_v.append((math.atan2(y, x)*360)/(2*math.pi))

    if len(face_detections) < 1:
        return None

    return landmarks_v


# TEST USE
# test_image_path = "test3.jpg"
# t = get_landmarks_from_face(test_image_path)
# print(t)

