import cv2

# path of pre-trained classifier
HAAR_CLASSIFIER_PATH = "haarcascade_frontalface_default.xml"

# face2 = cv2.CascadeClassifier("data/HAARCascades/haarcascade_frontalface_alt2.xml")
# face3 = cv2.CascadeClassifier("data/HAARCascades/haarcascade_frontalface_alt.xml")
# face4 = cv2.CascadeClassifier("data/HAARCascades/haarcascade_frontalface_alt_tree.xml")
#
# # Detecting faces
# face_2 = face2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
# face_3 = face3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
# face_4 = face4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)


def image_preprocess(image_path, output_path):
    image = cv2.imread(image_path)

    # Converts an image to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Loading the certain HAAR Cascade classifiers
    classifier = cv2.CascadeClassifier(HAAR_CLASSIFIER_PATH)
    face = classifier.detectMultiScale(gray, scaleFactor=1.1,
                                    minNeighbors=10,
                                    minSize=(5, 5),
                                    flags=cv2.CASCADE_SCALE_IMAGE)
    # Test output

    if len(face) == 1:
        req_face = face
    else:
        print "CANT FIND A FACE IN THAT IMAGE"
        return

    roi_gray = None
    for (x, y, w, h) in req_face:
        roi_gray = gray[y:y + h, x:x + w]

    # Writing the final image into the required path
    cv2.imwrite(output_path, cv2.resize(roi_gray, (350, 350)))


# TEST USE
image_preprocess("test.jpg", "res.jpg")
