from sklearn.externals import joblib
from get_face import *
from get_landmarks import *


def model_test(input_path):
    # Loading the SVM model from storage after training
    svm_classifer = joblib.load('models/svm4.pkl')

    image = cv2.imread(input_path)
    if image is None:
        print("CANT FIND THE IMAGE: %s!" % input_path)
        return

    get_face(image)

    pred_data = []
    landmarks_vec = get_landmarks_from_face(image)
    if landmarks_vec is None:
        return

    pred_data.append(landmarks_vec)
    np_pred_data = np.array(pred_data)
    res = svm_classifer.predict(np_pred_data)

    print(EMOTIONS[res[0]])
    cv2.putText(img=image,
                text=EMOTIONS[res[0]],
                org=(0, 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                lineType=2)

    cv2.imshow('result_image.jpg', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TEST USE

input_image_path = "test_image/anger.tiff"
# input_image_path = "test_image/happy.tiff"
# input_image_path = "test_image/sadness.tiff"
# input_image_path = "test_image/surprise.tiff"

model_test(input_image_path)

