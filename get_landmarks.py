from utils import *
from get_images_by_emotion import *
from get_landmarks_by_one_image import *


def get_landmarks_without_test():
    training_data = []
    training_labels = []

    for emotion in EMOTIONS:
        training_paths = get_training_images_by_emotion(emotion)
        for p in training_paths:
            image = cv2.imread(p)
            if image is None:
                print("CANT FIND THE IMAGE! %s" % p)
                return
            # Converts an image to gray
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # get the landmarks from one training image
            landmarks_vec = get_landmarks_from_face(gray_image)
            if landmarks_vec is not None:
                training_data.append(landmarks_vec)
                training_labels.append(EMOTIONS.index(emotion))
    return training_data, training_labels


def get_landmarks_with_test():
    training_data = []
    training_labels = []
    predicted_data = []
    predicted_labels = []

    for emotion in EMOTIONS:
        training_paths, predicted_paths = get_training_and_predicted_images_by_emotion(emotion)

        print("len: %d %d" % (len(training_paths), len(predicted_paths)))

        # get landmarks from training image
        for p in training_paths:
            image = cv2.imread(p)
            if image is None:
                print("CANT FIND THE IMAGE! %s" % p)
                return
            # Converts an image to gray
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # get the landmarks from one training image

            landmarks_vec = get_landmarks_from_face(gray_image)
            if landmarks_vec is None:
                pass
            else:
                training_data.append(landmarks_vec)
                training_labels.append(EMOTIONS.index(emotion))

        # test landmarks by test image
        for p in predicted_paths:
            image = cv2.imread(p)
            if image is None:
                print("CANT FIND THE IMAGE! %s" % p)
                return
            # Converts an image to gray
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # get the landmarks from one training image
            landmarks_vec = get_landmarks_from_face(gray_image)
            if landmarks_vec is None:
                pass
            else:
                predicted_data.append(landmarks_vec)
                predicted_labels.append(EMOTIONS.index(emotion))
        print(predicted_labels)

    return training_data, training_labels, predicted_data, predicted_labels


# TEST USE
# t, l1 = get_landmarks_without_test()
# print(t)
