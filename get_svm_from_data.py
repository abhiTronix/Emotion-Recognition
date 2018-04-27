from sklearn.externals import joblib
from sklearn.svm import SVC
from get_landmarks import *


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

    joblib.dump(svm_classifer, SVM_MODEL_PATH + "svm1.pkl")


def get_svm_model_with_test():
    # SVM classifier object
    svm_classifer = SVC(kernel='linear', probability=True, tol=1e-3)
    accuracy = []

    for i in range(0, 5):
        # GET LANDMARKS AND LABELS FROM ORGANIZED IMAGES
        training_data, training_labels, predicted_data, predicted_labels = get_landmarks_with_test()

        print("Test Set %d:" % (i+1))
        # LOAD INTO NUMPY ARRAY
        np_train_data = np.array(training_data)

        # Fit the SVM model according to the given training data
        svm_classifer.fit(np_train_data, training_labels)

        # Test THE SVM MODEL BY TEST IMAGES
        np_test_data = np.array(predicted_data)
        final_pred = svm_classifer.score(np_test_data, predicted_labels)
        print "Set %d: Matching percentage: %0.6f%%" % (i+1, final_pred*100)
        accuracy.append(final_pred)

    # SAVE THE SVM MODEL INTO DISK.
    joblib.dump(svm_classifer, SVM_MODEL_PATH + "svm4.pkl")


# TEST USE
# get_svm_model_with_test()
