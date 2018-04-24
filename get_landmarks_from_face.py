import dlib
import math
import numpy as np
import cv2

# Histogram equalization object
CLAHE = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))


def get_landmarks_from_face(face_detector, landmarks_predictor, image_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe_gray = CLAHE.apply(img_gray)
    face_detections = face_detector(clahe_gray, 1)
    landmarks_v = []

    for k, d in enumerate(face_detections):
        shape = landmarks_predictor(clahe_gray, d)
        x_cords = []
        y_cords = []
        for i in range(1, 68):
            x_cords.append(float(shape.part(i).x))
            y_cords.append(float(shape.part(i).y))

        x_mean = np.mean(x_cords)
        y_mean = np.mean(y_cords)

        # To compensate for variation in location of face in the frame.
        x_central = [(x - x_mean) for x in x_cords]
        y_central = [(y - y_mean) for y in y_cords]

        # 26 is the top of the bridge, 29 is the tip of the nose
        if x_cords[26] == x_cords[29]:
            anglenose=0
        else:
            anglenose_rad=int(math.atan((y_central[26] - y_central[29]) / (x_central[26] - x_central[29])))
            # Tan Inverse of slope
            anglenose=int(math.degrees(anglenose_rad))
            # print(y_central[26]-y_central[29])
            # print(y_cords[26]-y_cords[29])

        if anglenose<0:
            anglenose+=90 # Because anglenose computed above is the angle wrt to vertical

        else:
            anglenose-=90      # Because anglenose computed above is the angle wrt to vertical

        for x,y,w,z in zip(x_central,y_central,x_cords,y_cords):
            landmarks_v.append(x) # Co-ordinates are added relative to the Centre of gravity of face to accompany for
            landmarks_v.append(y) # variation of location of face in the image.

            # Euclidean distance between each point and the centre point (length of vector)
            np_mean_coor = np.asarray((y_mean, x_mean))
            np_coor = np.asarray((z, w))
            euclid_d = np.linalg.norm(np_coor-np_mean_coor)
            landmarks_v.append(euclid_d)

            # Angle of the vector, which is used to correct for the offset caused due to tilt of image wrt horizontal
            angle_rad = (math.atan((z - y_mean) / (w - x_mean)))
            angle_degree = math.degrees(angle_rad)
            angle_req = int(angle_degree - anglenose)
            landmarks_v.append(angle_req)

    if len(face_detections) < 1:
        return None

    return landmarks_v


# TEST USE
face_detector = dlib.get_frontal_face_detector()
landmarks_predictor = dlib.shape_predictor("data/dlib_data/shape_predictor_68_face_landmarks.dat")
test_image_path = "source_images/anger/S010_005_01593512.png"
t = get_landmarks_from_face(face_detector, landmarks_predictor, test_image_path)
print t

