from keras.datasets import cifar10
import matplotlib
import numpy

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)
