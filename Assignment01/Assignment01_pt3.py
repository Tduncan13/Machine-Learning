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
x_train = train_images_original.reshape(count, dim, 1)
x_train = x_train.astype('float32') / 255 
x_test = test_images_original.reshape(10000, dim)
x_test = x_test.astype('float32') / 255

# Convert Labels in to One-Hot Vectors
y_train = to_categorical(train_labels_original)
y_test = to_categorical(test_labels_original)

# Initialize Weight matrix
# np.random.seed(42)
initial_weight = np.zeros((dim, 10))
w = initial_weight

# Initialize Bias vector
initial_bias = np.random.rand(10, 1)
b = initial_bias
# Initialize Kronecker Delta
delta = np.identity(10)
epsilon = 1 / 922337203685477

# Define Softmax
def softmax(z):
    return np.exp(z) / np.exp(z).sum()

def cce(y, a):
    return  -1 * (y.T.dot(np.log10(a))) 

def cce_prime_w(x, y, a, d):
    return x * y.T.dot(a - d)

def cce_prime_b(y, a, d):
    return y.T.dot(a - d)

# Train the model. 
def train_model():
    global w
    global b
    learning_rate = 0.1
    for i in range(count):
        x = x_train[i].reshape(dim, 1)
        y = y_train[i].reshape(10, 1)
        z = (w.T.dot(x) + b) / dim
        a = softmax(z)
        w_gradient = cce_prime_w(x, y, a, delta)
        b_gradient = cce_prime_b(y, a, delta).T 
        w = w - (w_gradient * learning_rate) 
        b = b - (b_gradient * learning_rate) 
        # print(np.exp(z).sum())
       

# Test the model. 
def test_model():
    global w
    global b
    loss = 0.0
    acc = 0.0
    for i in range(10000):
        x = x_test[i].reshape(dim, 1)
        y = y_test[i].reshape(10, 1)
        z = (w.T.dot(x) + b) / dim
        a = softmax(z)
        loss += cce(y, a)
        acc += a[np.argmax(y)]

    print("Loss - %0.4f" % (loss / 10000))
    print("Accuracty - %0.4f" % (acc / 10000))
        

train_model()
print("Model Trained!")

test_model()
