import glob
import random
import os
from utils import ORGANIZED_IMAGES_PATH


def get_training_images_by_emotion(emotion):
    return glob.glob(ORGANIZED_IMAGES_PATH + "/" + emotion + "/*")


def get_training_and_predicted_images_by_emotion(emotion):
    image_paths = glob.glob(ORGANIZED_IMAGES_PATH + "/" + emotion + '/*')
    # Shuffle the list of image paths
    random.shuffle(image_paths)

    segment = int(len(image_paths)*0.90)

    # 90 percent of images are used for training data
    training_image_paths = image_paths[:segment]
    
    # 10 percent of images are used as test data
    test_image_paths = image_paths[segment:]

    return training_image_paths, test_image_paths


# TEST USE
# p1, p2 = get_images_by_emotion("anger")
# print p1
# print p2
