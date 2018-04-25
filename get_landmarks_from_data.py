from sklearn.externals import joblib
from sklearn.svm import SVC
from utils import *
from get_images_by_emotion import *
from get_landmarks_by_one_image import *


def get_all_landmarks_by_data():
    training_data = []
    training_labels = []

    for emotion in EMOTIONS:
        training_paths, predicted_paths = get_images_by_emotion(emotion)

        # get landmarks from training image
        for p in training_paths:
            landmarks_vec = get_landmarks_from_face(p)
            if landmarks_vec is None:
                pass
            else:
                training_data.append(landmarks_vec)
                training_labels.append(EMOTIONS.index(emotion))

    return training_data, training_labels


def get_landmarks_without_test():
    training_data = []
    training_labels = []

    for emotion in EMOTIONS:
        training_paths = get_training_images_by_emotion(emotion)
        for p in training_paths:
            landmarks_vec = get_landmarks_from_face(p)
            if landmarks_vec is None:
                pass
            else:
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

        # get landmarks from training image
        for p in training_paths:
            landmarks_vec = get_landmarks_from_face(p)
            if landmarks_vec is None:
                pass
            else:
                training_data.append(landmarks_vec)
                training_labels.append(EMOTIONS.index(emotion))

        # test landmarks by test image
        for p in predicted_paths:
            landmarks_vec = get_landmarks_from_face(p)
            if landmarks_vec is None:
                pass
            else:
                predicted_data.append(landmarks_vec)
                predicted_labels.append(EMOTIONS.index(emotion))

    return training_data, training_labels, predicted_data, predicted_labels


def get_svm_model_without_test():
    # INITIALIZE THE SVM MODEL
    svm_classifer = SVC(kernel='linear', probability=True, tol=1e-3)

    # GET LANDMARKS AND LABELS FROM ORGANIZED IMAGES
    training_data, training_labels = get_landmarks_without_test()

    # LOAD INTO NUMPY ARRAY
    np_train_data = np.array(training_data)
    np_train_labels = np.array(training_labels)

    # Fit the SVM model according to the given training data
    svm_classifer.fit(np_train_data, np_train_labels)


def get_svm_model_with_test():
    # SVM classifier object
    svm_classifer = SVC(kernel='linear', probability=True, tol=1e-3)

    accuracy = []

    # GET LANDMARKS AND LABELS FROM ORGANIZED IMAGES
    training_data, training_labels, predicted_data, predicted_labels = get_landmarks_with_test()

    # LOAD INTO NUMPY ARRAY
    np_train_data = np.array(training_data)
    np_train_labels = np.array(training_labels)

    # Fit the SVM model according to the given training data
    svm_classifer.fit(np_train_data, np_train_labels)

    # Test THE SVM MODEL BY TEST IMAGES
    np_test_data = np.array(predicted_data)
    final_pred = svm_classifer.score(np_test_data, predicted_labels)
    accuracy.append(final_pred)

    # SAVE THE SVM MODEL INTO DISK.
    joblib.dump(svm_classifer, SVM_MODEL_PATH)


# TEST USE
get_svm_model_without_test()
