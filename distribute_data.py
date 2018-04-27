from utils import *
import glob
from shutil import copy2
import os


def distribute():
    image_paths = glob.glob("jaffe/*")
    if not os.path.exists("organized_images/surprise/"):
        os.makedirs("organized_images/surprise/")
    if not os.path.exists("organized_images/sadness/"):
        os.makedirs("organized_images/sadness/")
    if not os.path.exists("organized_images/neutral/"):
        os.makedirs("organized_images/neutral/")
    if not os.path.exists("organized_images/happy/"):
        os.makedirs("organized_images/happy/")
    if not os.path.exists("organized_images/anger/"):
        os.makedirs("organized_images/anger/")

    for path in image_paths:
        filename = os.path.basename(path)
        filename_list = filename.split(".")
        if len(filename_list) < 4:
            pass

        if filename_list[1].startswith("SU"):
            copy2(path, "organized_images/surprise/" + filename)
        elif filename_list[1].startswith("SA"):
            copy2(path, "organized_images/sadness/" + filename)
        elif filename_list[1].startswith("NE"):
            copy2(path, "organized_images/neutral/" + filename)
        elif filename_list[1].startswith("HA"):
            copy2(path, "organized_images/happy/" + filename)
        elif filename_list[1].startswith("AN"):
            copy2(path, "organized_images/anger/" + filename)


# TEST USE
distribute()





