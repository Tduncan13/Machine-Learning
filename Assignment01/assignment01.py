import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

#==========================================================
#  Written by Tyler Duncan
#  Date 02/13/2019
#
#  The purpose of this program is to create a Neural 
#  Network that can classify images of handwritten 
#  digits using only numpy.  This is part 1 of a 5 
#  part problem set in Dr. Pawel Wocjan's Machine 
#  Learning course at the University of Central Florida
#==========================================================



# Retrieve data set. 
mnist = tf.keras.datasets.mnist 

# Unpack dataset. 
(train_images_original, train_labels_original), (test_images_original, test_labels_original) = mnist.load_data()
count, rows, cols = train_images_original.shape

dim = rows * cols

# Reshape into input vectors. 
train_images = train_images_original.reshape(count, dim)
train_images = train_images.astype('float32') / 255 
test_images = test_images_original.reshape(10000, dim)
test_images = test_images.astype('float32') / 255

# Convert Labels in to One-Hot Vectors
train_labels = to_categorical(train_labels_original)
test_labels = to_categorical(test_labels_original)

# Initialize Weight vector
np.random.seed(42)
initial_weight = np.random.randn(dim, 1)


