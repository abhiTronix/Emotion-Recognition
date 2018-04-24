import glob
import random

# PATH OF DIRECTORY OF TRAINING IMAGES
IMAGE_PATH = "source_images/"


def get_images_by_emotion(emotion):
    image_paths = glob.glob(IMAGE_PATH + emotion + '/*')
    # Shuffle the list of image paths
    random.shuffle(image_paths)

    # 90 percent of images are used for training data
    training_image_paths = image_paths[:int(len(image_paths)*0.90)]
    
    # 10 percent of images are used as test data
    test_image_paths = image_paths[-int(len(image_paths)*0.10):]
    
    return training_image_paths, test_image_paths


# TEST USE
p1, p2 = get_images_by_emotion("anger")
print p1
print p2
