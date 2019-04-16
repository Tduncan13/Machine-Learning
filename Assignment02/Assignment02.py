from keras.datasets import cifar10
import matplotlib
import numpy

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

train_images.shape
