import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2


# load image dataset from dataset/train and dataset/test
train_path = 'dataset/train'
test_path = 'dataset/test'


def load_data(path):
    images = []
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            image = cv2.imread(path + '/' + filename)
            images.append(image)
    return images


train_images = load_data(train_path)
test_images = load_data(test_path)
print("Done loading images.")

# spliting each image into half


def split_image(images):
    original = []
    mask = []
    for image in images:
        original.append(image[:, :int(image.shape[1]/2), :])
        mask.append(image[:, int(image.shape[1]/2):, :])
    return original, mask


train_original, train_mask = split_image(train_images)
test_original, test_mask = split_image(test_images)

# k-means clustering on images
