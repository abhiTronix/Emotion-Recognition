from utils import *
import numpy as np
import dlib
import copy


def get_face(origin_image):
    if origin_image is None:
        print("CANT FIND THE IMAGE!")
        return

    # Converts an image to gray
    gray = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)

    # Loading the certain HAAR Cascade classifiers
    faceCascade1 = cv2.CascadeClassifier(HAAR_CLASSIFIER_PATH_1)
    faceCascade2 = cv2.CascadeClassifier(HAAR_CLASSIFIER_PATH_2)
    faceCascade3 = cv2.CascadeClassifier(HAAR_CLASSIFIER_PATH_3)
    faceCascade4 = cv2.CascadeClassifier(HAAR_CLASSIFIER_PATH_4)

    face1 = faceCascade1.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face2 = faceCascade2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face3 = faceCascade3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face4 = faceCascade4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Test output
    if len(face1) == 1:
        req_face = face1
    elif len(face2) == 1:
        req_face = face2
    elif len(face3) == 1:
        req_face = face3
    elif len(face4) == 1:
        req_face = face4
    else:
        print("CANT FIND A FACE IN THAT IMAGE")
        return

    roi_gray = None
    for (x, y, w, h) in req_face:
        roi_gray = gray[y:y + h, x:x + w]

    if roi_gray is None:
        return None

    # Writing the final image into the required path
    resized_image = cv2.resize(roi_gray, (500, 500))
    cv2.imwrite("res1.jpg", resized_image)

    # GET THE LANDMARKS FROM IMAGE
    detector = dlib.get_frontal_face_detector()  # Face detector
    predictor = dlib.shape_predictor("data/dlib_data/shape_predictor_68_face_landmarks.dat")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)
    detections = detector(clahe_image, 1)  # Detect the faces in the image

    # PRINT THE LANDMARKS IN IMAGE
    landmarks = copy.deepcopy(origin_image)
    for k, d in enumerate(detections):  # For each detected face
        shape = predictor(clahe_image, d)  # Get coordinates
        for i in range(2, 68):  # There are 68 landmark points on each face
            # For each point, draw a red circle with thickness2 on the original frame
            x = shape.part(i).x
            y = shape.part(i).y
            cv2.circle(img=landmarks,
                       center=(x, y),
                       radius=1,
                       color=(255, 0, 0),
                       thickness=2)
    cv2.imwrite("res2.jpg", landmarks)   # SAVE ALL LANDMARKS

    # PRINT THE CENTER POINT IN IMAGE
    center = copy.deepcopy(origin_image)
    x_list = []
    y_list = []
    for k, d in enumerate(detections):
        shape = predictor(clahe_image, d)
        for i in range(1, 68):
            x_list.append(float(shape.part(i).x))
            y_list.append(float(shape.part(i).y))

    x_mean = np.mean(x_list)
    y_mean = np.mean(y_list)
    cv2.circle(img=center,
               center=(int(x_mean), int(y_mean)),
               radius=1,
               color=(255, 0, 0),
               thickness=4)
    cv2.imwrite("res3.jpg", center)  # SAVE THE CENTER POINT

    # PRINT THE LINES BETWEEN LANDMARKS AND CENTER POINT
    lines = copy.deepcopy(origin_image)
    for k, d in enumerate(detections):  # For each detected face
        shape = predictor(clahe_image, d)  # Get coordinates
        for i in range(2, 68):  # There are 68 landmark points on each face
            # For each point, draw a red circle with thickness2 on the original frame
            x = shape.part(i).x
            y = shape.part(i).y
            cv2.line(lines, (x, y), (int(x_mean), int(y_mean)), (255, 0, 0), 2)
    cv2.imwrite("res4.jpg", lines)  # SAVE THE LINES


# TEST USE
image = cv2.imread("test_image/anger.tiff")
get_face(image)
