import os
import boto3
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import time
import json
import shutil
from IPython.display import clear_output
import numpy as np 
from tqdm import tqdm
import cv2
import os
import shutil
import itertools
import imutils
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
import sagemaker.amazon.common as smac
from sagemaker import get_execution_role


role = get_execution_role()


def load_data(dir_path, img_size=(100,100)):
    """
    Load resized images as np.arrays to workspace
    """
    X = []
    y = []
    i = 0
    labels = dict()
    for path in tqdm(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            labels[i] = path
            for file in os.listdir(dir_path + path):
                if not file.startswith('.'):
                    img = cv2.imread(dir_path + path + '/' + file)
                    X.append(img)
                    y.append(i)
            i += 1
    X = np.array(X)
    y = np.array(y)
    print(f'{len(X)} images loaded from {dir_path} directory.')
    return X, y, labels

def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)

def split_load_crop(img_path, train_dir, test_dir, val_dir, img_size):
    # split the data by train/val/test
    for CLASS in os.listdir(img_path):
        if not CLASS.startswith('.'):
            IMG_NUM = len(os.listdir(img_path + CLASS))
            for (n, FILE_NAME) in enumerate(os.listdir(img_path + CLASS)):
                img = img_path + CLASS + '/' + FILE_NAME
                if n < 5:
                    shutil.copy(img, test_dir + CLASS.upper() + '/' + FILE_NAME)
                elif n < 0.8*IMG_NUM:
                    shutil.copy(img, train_dir + CLASS.upper() + '/' + FILE_NAME)
                else:
                    shutil.copy(img, val_dir + CLASS.upper() + '/' + FILE_NAME)
    # use predefined function to load the image data into workspace
    X_train, y_train, labels = load_data(train_dir, img_size)
    X_test, y_test, _ = load_data(test_dir, img_size)
    X_val, y_val, _ = load_data(val_dir, img_size)
    X_train_crop = crop_imgs(set_name=X_train)
    X_val_crop = crop_imgs(set_name=X_val)
    X_test_crop = crop_imgs(set_name=X_test)
    save_new_images(X_train_crop, y_train, folder_name='TRAIN_CROP/')
    save_new_images(X_val_crop, y_val, folder_name='VAL_CROP/')
    save_new_images(X_test_crop, y_test, folder_name='TEST_CROP/')


def save_new_images(x_set, y_set, folder_name):
    i = 0
    for (img, imclass) in zip(x_set, y_set):
        if imclass == 0:
            cv2.imwrite(folder_name+'NO/'+str(i)+'.jpg', img)
        else:
            cv2.imwrite(folder_name+'YES/'+str(i)+'.jpg', img)
        i += 1

def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-15 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(
            img,
            dsize=img_size,
            interpolation=cv2.INTER_CUBIC
        )
        set_new.append(preprocess_input(img))
    return np.array(set_new)


