import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
from sklearn.cluster import KMeans
from time import time
from sklearn import metrics
from tensorflow.keras import layers
from focal_loss import BinaryFocalLoss

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

# spliting each image into half and stacking them into a single array


def split_image(images):
    original = []
    mask = []
    for image in images:
        original.append(image[:, :int(image.shape[1]/2), :])
        mask.append(image[:, int(image.shape[1]/2):, :])
    return original, mask


train_original, train_mask = split_image(train_images)
test_original, test_mask = split_image(test_images)

# k-means clustering on images for aquiring cluster centers


def kmeans(image, flag):
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)  # criteria
    k = 15  # Choosing number of cluster
    pixel_vals = image
    if (flag):
        pixel_vals = image.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    retval, labels, centers = cv2.kmeans(
        pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return labels, centers

# perform k-means for final clusters on all centers


def clustering(images):
    labels = []
    centers = [[0, 0, 0]]
    for image in images:
        l, c = kmeans(image, True)
        centers = np.concatenate((centers, c), axis=0)
        labels.append(l)
    t, cts = kmeans(centers[1:], False)
    return labels, cts


def eucl_distance(centroid, pixel):
    distance = 0
    for i in range(len(centroid)):
        distance += np.square(int(pixel[i]) - int(centroid[i]))
    distance = np.sqrt(distance)
    return distance


def image_labels(centroids, pixels):
    labels = np.zeros(len(pixels))

    for i in range(len(pixels)):
        distance = 2**10
        choosen_cluster = -1

        for j in range(len(centroids)):
            if(eucl_distance(centroids[j], pixels[i]) < distance):
                distance = eucl_distance(centroids[j], pixels[i])
                choosen_cluster = j
                labels[i] = j
    return labels

# change pixel to cluster center for all images


def pixel_to_centroids(images, centroids):
    new_images = []
    for image in images:
        pixels = image.reshape((-1, 3))
        pixels = np.float32(pixels)
        labels = image_labels(centroids, pixels)
        cents = np.uint8(centroids)
        int_lbls = []
        for label in labels:
            int_lbls.append(math.trunc(label))
        pixel_vals = cents[int_lbls]
        # reshape data into the original image dimensions
        segmented_image = pixel_vals.reshape((images[0].shape))
        plt.imshow(segmented_image)
        new_images.append(segmented_image)
    return new_images


train_images = load_data(train_path)
test_images = load_data(test_path)
print("Done loading images.")
train_original, train_mask = split_image(train_images)
test_original, test_mask = split_image(test_images)
print("Done spliting images.")
train_o = np.array(train_original)
test_o = np.array(test_original)
# masks = np.concatenate((train_mask, test_mask), axis=0)

centroids = [[157.77898, 147.46292, 178.24544],
             [186.2352,  58.806168, 197.36371],
             [17.504044,  10.361008,  15.13903],
             [127.34017,   9.706383,  11.866598],
             [176.13904, 170.30707, 234.36456],
             [130.63435,  59.152813, 128.74097],
             [60.830353,  31.615223, 185.01045],
             [131.33203, 150.2643, 150.03224],
             [172.7652, 128.60927,  75.4259],
             [164.99858, 235.18184, 169.15973],
             [117.62971, 111.08022, 101.388794],
             [48.97888, 200.55396, 208.70932],
             [61.91623,  56.814667,  77.04306],
             [50.203438, 137.38998, 109.651474],
             [225.85764,  40.00203, 237.25241]]

# labels, centroids = clustering(train_mask)
# segmented = pixel_to_centroids(train_mask, centroids)
# segmented_test = pixel_to_centroids(test_mask, centroids)
