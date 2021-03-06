import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import random
import numba 

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

#==============================================================
#  For problem 5 read up on Connected Components Alg using DFS
#==============================================================

# Retrieve data set. 
mnist = tf.keras.datasets.mnist 

# Unpack dataset. 
(train_images_original, train_labels_original), (test_images_original, test_labels_original) = mnist.load_data()
count, rows, cols = train_images_original.shape

dim = rows * cols


# Reshape into input vectors. 
train_images = train_images_original.reshape(count, dim, 1)
train_images = train_images.astype('float32') / 255 
test_images = test_images_original.reshape(10000, dim)
test_images = test_images.astype('float32') / 255

# Convert Labels in to One-Hot Vectors
train_labels = to_categorical(train_labels_original)
test_labels = to_categorical(test_labels_original)


# Define Classifier. 
class Classifier: 
    # Construct the Classifier with Dataset. 
    def __init__(self, tri, trl, tsti, tstl, number):
        self.tri = tri
        self.trl = trl
        self.tsti = tsti 
        self.tstl = tstl
        self.number = number

    # Initialize Weight vector
    np.random.seed(42)
    initial_weight = np.random.randn(dim, 1)

    # Initialize Bias
    initial_bias = np.random.randint(0, 1, size=None)

    # Initialize Weights and Bias
    w = initial_weight
    b = initial_bias

    # Define the sigmoid funciton. 
   
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Define sigmoid prime. 
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    # Define Squared Error Loss
    def se(self, y, a):
        return 0.5 * ((a - y) ** 2)

    # Define Squared Error Loss Prime
    def se_prime(self, a, y):
        return a - y

    # Define Binary Cross Entropy Loss.
    def bce(self, y, a):
        return (-y * np.log10(a)) - ((1 - y) * np.log10(1 - a))

    # Everything simplifies to (a - y)x_j for Binary Cross Entropy's derivative. 
    def bce_prime(self, y, a, x):
        return x * np.subtract(a, y)

    def bce_prime_b(self, y, a):
        return a - y

    
    def train_model_bce(self):
        # Set Learning rate and batch size. 
        learning_rate = 0.2
        batch_size = 32
        epochs = 10

        for epoch in range(epochs):
            # Train the network. 
            for i in range(0, count, batch_size):
                # Shuffle Training Set. 
                s = np.arange(self.tri.shape[0])
                np.random.shuffle(s)
                train_image_shuffled = self.tri[s]
                train_label_shuffled = self.trl[s]
                train_label_batch = train_label_shuffled[i:i+batch_size]
                x = train_image_shuffled[i:i+batch_size].reshape((32, 784)).T
                y = train_label_batch[:,self.number].reshape((32,1)).T
                z = (self.w.T.dot(x) + self.b)
                a = self.sigmoid(z)
                print(self.b)
                loss = self.bce(y, a) 
                print(loss)  
                gradient_w = (( 1 / batch_size) * np.sum(self.bce_prime(y, a, x), 1)) * learning_rate * -1
                gradient_b = (( 1 / batch_size) * np.sum(self.bce_prime_b(y, a), 1)) * learning_rate * -1
                self.w = self.w + gradient_w.reshape(dim, 1)
                self.b += ((1 / batch_size) * np.sum(gradient_b))
                # b = self.bce_prime_b(y, a)

    def make_prediction(self, ti):
        x = ti
        z = np.dot(self.w.T, x) + self.b
        return self.sigmoid(z)
        
# Create models.
model_zero = Classifier(train_images, train_labels, test_images, test_labels, 0)
model_one = Classifier(train_images, train_labels, test_images, test_labels, 1)
model_two = Classifier(train_images, train_labels, test_images, test_labels, 2)
model_three = Classifier(train_images, train_labels, test_images, test_labels, 3)
model_four = Classifier(train_images, train_labels, test_images, test_labels, 4)
model_five = Classifier(train_images, train_labels, test_images, test_labels, 5)
model_six = Classifier(train_images, train_labels, test_images, test_labels, 6)
model_seven = Classifier(train_images, train_labels, test_images, test_labels, 7)
model_eight = Classifier(train_images, train_labels, test_images, test_labels, 8)
model_nine = Classifier(train_images, train_labels, test_images, test_labels, 9)

# Train Models
model_zero.train_model_bce()
# model_one.train_model_bce()
# model_two.train_model_bce()
# model_three.train_model_bce()
# model_four.train_model_bce()
# model_five.train_model_bce()
# model_six.train_model_bce()
# model_seven.train_model_bce()
# model_eight.train_model_bce()
# model_nine.train_model_bce()

# prediciton_vector = []
# q = 5100
# # Test Models. 
# for q in range(0, count):
#     prediciton_vector.append(model_zero.make_prediction(test_images[q]))
#     prediciton_vector.append(model_one.make_prediction(test_images[q]))
#     prediciton_vector.append(model_two.make_prediction(test_images[q]))
#     prediciton_vector.append(model_three.make_prediction(test_images[q]))
#     prediciton_vector.append(model_four.make_prediction(test_images[q]))
#     prediciton_vector.append(model_five.make_prediction(test_images[q]))
#     prediciton_vector.append(model_six.make_prediction(test_images[q]))
#     prediciton_vector.append(model_seven.make_prediction(test_images[q]))
#     prediciton_vector.append(model_eight.make_prediction(test_images[q]))
#     prediciton_vector.append(model_nine.make_prediction(test_images[q]))

# test = np.asarray(prediciton_vector)
# print(np.argmax(test))
# print(np.argmax(test_labels[q]))