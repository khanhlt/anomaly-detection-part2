import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

my_path = os.path.abspath(os.path.dirname(__file__))
train_folder = os.path.join(my_path, "dataset/ok_data")
test_folder = os.path.join(my_path, "dataset/ng_data")

def read_image_flatten(folder):
    res = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename),0)
        img = np.asarray(img).astype('float32')/255.
        img = img.flatten()
        res.append(img)
    return np.asarray(res)


def read_image(folder):
    res = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename),0)
        res.append(img)
    return np.asarray(res)

def read_image_and_label(folder):
    res = []
    label = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), 0)
        img = np.asarray(img).astype('float32') / 255.
        img = img.flatten()
        res.append(img)
        label.append(filename)
    return np.asarray(res), np.asarray(label)

def load_data_origin():
    train = read_image_flatten(train_folder)
    test = read_image_flatten(test_folder)
    return train, test

def load_data():
    train = read_image(train_folder)
    x_train, test_norm = train_test_split(train, test_size=0.2, random_state=10)
    test = read_image(test_folder)
    num_test_anom = len(test)
    num_test_norm = len(test_norm)
    test_label = np.ones(len(test))
    test = np.concatenate((test, test_norm), axis=0)
    test_label = np.concatenate((test_label, np.zeros(len(test_norm))), axis=0)

    return x_train, test, test_label, num_test_norm, num_test_anom