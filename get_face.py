from utils import *
import dlib


def get_face(origin_image):
    if origin_image is None:
        print "CANT FIND THE IMAGE!"
        return

    # Converts an image to gray
    gray = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)

    # Loading the certain HAAR Cascade classifiers
    faceCascade1 = cv2.CascadeClassifier(HAAR_CLASSIFIER_PATH_1)
    faceCascade2 = cv2.CascadeClassifier(HAAR_CLASSIFIER_PATH_2)
    faceCascade3 = cv2.CascadeClassifier(HAAR_CLASSIFIER_PATH_3)
    faceCascade4 = cv2.CascadeClassifier(HAAR_CLASSIFIER_PATH_4)

    face1 = faceCascade1.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
    face2 = faceCascade2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
    face3 = faceCascade3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
    face4 = faceCascade4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

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
        print "CANT FIND A FACE IN THAT IMAGE"
        return

    roi_gray = None
    for (x, y, w, h) in req_face:
        roi_gray = gray[y:y + h, x:x + w]

    if roi_gray is None:
        return None

    # Writing the final image into the required path
    cv2.imwrite("res1.jpg", cv2.resize(roi_gray, (500, 500)))

    detector = dlib.get_frontal_face_detector()  # Face detector
    predictor = dlib.shape_predictor("data/dlib_data/shape_predictor_68_face_landmarks.dat")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)

    detections = detector(clahe_image, 1)  # Detect the faces in the image

    for k, d in enumerate(detections):  # For each detected face
        shape = predictor(clahe_image, d)  # Get coordinates
        for i in range(1, 68):  # There are 68 landmark points on each face
            # For each point, draw a red circle with thickness2 on the original frame
            cv2.circle(img=origin_image,
                       center=(shape.part(i).x, shape.part(i).y),
                       radius=1,
                       color=(255, 255, 255),
                       thickness=2)
    cv2.imwrite("res2.jpg", origin_image)  # save the frame


# TEST USE
# get_face("sad_face.jpg", "res.jpg")
