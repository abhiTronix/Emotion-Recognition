import cv2

# normal, sans-serif
FONT_TYPE = cv2.FONT_HERSHEY_DUPLEX

# PATH OF TRAINED SVM MODEL
SVM_MODEL_PATH = "models/"

# CATEGORIES OF EMOTIONS
EMOTIONS = ["anger", "happy", "sadness", "surprise"]

# PATH OF DIRECTORY OF TRAINING IMAGES
# ORGANIZED_IMAGES_PATH = "organized_images"
ORGANIZED_IMAGES_PATH = "organized_images"

# HISTOGRAM EQUALIZATION OBJECT
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# PATH OF PRE-TRAINED CLASSIFIER
HAAR_CLASSIFIER_PATH_1 = "data/opencv_data/haarcascade_frontalface_default.xml"
HAAR_CLASSIFIER_PATH_2 = "data/opencv_data/haarcascade_frontalface_alt.xml"
HAAR_CLASSIFIER_PATH_3 = "data/opencv_data/haarcascade_frontalface_alt2.xml"
HAAR_CLASSIFIER_PATH_4 = "data/opencv_data/haarcascade_frontalface_alt_tree.xml"

# PATH OF LANDMARKS PREDICTOR
LANDMARKS_PREDICTOR_PATH = "data/dlib_data/shape_predictor_68_face_landmarks.dat"
