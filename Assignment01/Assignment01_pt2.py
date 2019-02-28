import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import random

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
    weights = initial_weight
    bias = initial_bias

    # Define the sigmoid funciton. 
    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    # Define sigmoid prime. 
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))


    # Define Binary Cross Entropy Loss.
    def bce(self, y, a):
        return (-y * np.log(a)) - ((1 - y) * np.log(1 - a))

    # Everything simplifies to (a - y)x_j for Binary Cross Entropy's derivative. 
    def bce_prime(self, y, a, x):
        return np.multiply(x, a - y)

    def bce_prime_bias(self, y, a):
        return a - y

    def train_model_bce(self):
        learning_rate = 0.5
        # Train the network. 
        for i in range(count):
            x = self.tri[i]
            z = np.dot(self.weights.T, x) + self.bias
            a = self.sigmoid(z)
            y = int(np.argmax(self.trl[i]) == self.number) 
            gradient_w = np.multiply((learning_rate * -1), self.bce_prime(y, a, x))
            gradient_b = self.bce_prime_bias(y, a) * learning_rate * -1
            self.weights = np.add(self.weights, gradient_w)
            self.bias += gradient_b

    def make_prediction(self, ti):
        x = ti
        z = np.dot(self.weights.T, x) + self.bias
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
model_one.train_model_bce()
model_two.train_model_bce()
model_three.train_model_bce()
model_four.train_model_bce()
model_five.train_model_bce()
model_six.train_model_bce()
model_seven.train_model_bce()
model_eight.train_model_bce()
model_nine.train_model_bce()

correct = 0
# Test Models. 
for q in range(10000):
    prediciton_vector = []
    prediciton_vector.append(model_zero.make_prediction(test_images[q]))
    prediciton_vector.append(model_one.make_prediction(test_images[q]))
    prediciton_vector.append(model_two.make_prediction(test_images[q]))
    prediciton_vector.append(model_three.make_prediction(test_images[q]))
    prediciton_vector.append(model_four.make_prediction(test_images[q]))
    prediciton_vector.append(model_five.make_prediction(test_images[q]))
    prediciton_vector.append(model_six.make_prediction(test_images[q]))
    prediciton_vector.append(model_seven.make_prediction(test_images[q]))
    prediciton_vector.append(model_eight.make_prediction(test_images[q]))
    prediciton_vector.append(model_nine.make_prediction(test_images[q]))
    test = np.asarray(prediciton_vector)
    if np.argmax(test) == np.argmax(test_labels[q]):
        correct += 1

print("Accuracy - %0.4f" % ((correct * 100 ) / 10000))