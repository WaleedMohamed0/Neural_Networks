import os
from random import shuffle
import cv2
import numpy as np
from tqdm import tqdm

IMG_SIZE = 120
LR = 1e-3
TRAIN_DIR = 'Train/'
TEST_DIR = 'Test/'


def convertToJPG(path):
    for i in os.listdir(path):
        os.rename(os.path.join(path, i), os.path.join(path, os.path.splitext(i)[0] + '.jpg'))


def create_label(image_name):
    if image_name.__contains__('Basketball'):
        return np.array([1, 0, 0, 0, 0, 0])
    elif image_name.__contains__('Football'):
        return np.array([0, 1, 0, 0, 0, 0])
    elif image_name.__contains__('Rowing'):
        return np.array([0, 0, 1, 0, 0, 0])
    elif image_name.__contains__('Swimming'):
        return np.array([0, 0, 0, 1, 0, 0])
    elif image_name.__contains__('Tennis'):
        return np.array([0, 0, 0, 0, 1, 0])
    elif image_name.__contains__('Yoga'):
        return np.array([0, 0, 0, 0, 0, 1])
    else:
        print("Error in label creation")


def create_data(dir):
    data = []
    for img in tqdm(os.listdir(dir)):
        path = os.path.join(dir, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append([np.array(img), create_label(path)])
    shuffle(data)
    np.save('train_data.npy', data)
    return data


def augment_data(dir):
    data = []
    for img in tqdm(os.listdir(dir)):
        path = os.path.join(dir, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append([np.array(img), create_label(path)])
        data.append([np.array(cv2.flip(img, 1)), create_label(path)])
        data.append([np.array(cv2.flip(img, -1)), create_label(path)])
        data.append([np.array(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)), create_label(path)])
        data.append([np.array(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)), create_label(path)])
        data.append([np.array(cv2.rotate(img, cv2.ROTATE_180)), create_label(path)])
    shuffle(data)
    np.save('drive/My Drive/Colab Notebooks/train_data.npy', data)
    return data


