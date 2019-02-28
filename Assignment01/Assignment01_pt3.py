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

# Set dimension constant. 
dim = rows * cols

# Reshape into input vectors. 
x_train = train_images_original.reshape(count, dim)
x_train = x_train.astype('float32') / 255 
x_test = test_images_original.reshape(10000, dim)
x_test = x_test.astype('float32') / 255

# Convert Labels in to One-Hot Vectors
y_train = to_categorical(train_labels_original)
y_test = to_categorical(test_labels_original)

# Initialize Weight matrix
initial_weight = np.zeros((dim, 10))
w = initial_weight

# Initialize Bias vector
initial_bias = np.random.rand(10, 1)
b = initial_bias

# Define Softmax
def softmax(z):
    z1 = z.max()
    z2 = np.exp(z - z1)
    return z2 / z2.sum()

# Define Categorical Cross-Entropy
def cce(y, z):
    return  -1 * np.sum(y.T.dot(z - np.log(np.exp(z).sum()))) 

# Define CCE's derivative with respect to w. 
def cce_prime_w(x, y, a):
    dz = a - y.T 
    return x.T.dot(dz.T)

# Define CCE's derivative with respect to b. 
def cce_prime_b(y, a):
    return np.sum(a - y.T)

# Train the model using Mini-Batch Stochastic Gradient Descent. 
def train_model_mini_batch(epochs, batch_size, learning_rate):
    global w
    global b
    
    for epoch in range(epochs):
        s = np.random.permutation(count)
        x_shuffled = x_train[s]
        y_shuffled = y_train[s]
        for i in range(0, count, batch_size):
            x = x_shuffled[i:i+batch_size]
            y = y_shuffled[i:i+batch_size]
            z = w.T.dot(x.T) + b 
            a = softmax(z)
            dw = cce_prime_w(x, y, a) * (1 / batch_size)
            db = cce_prime_b(y, a).T * (1 / batch_size)
            w = w - (dw * learning_rate) 
            b = b - (db * learning_rate) 
            
        print("Finished Epoch %d" % epoch)

# Test the model. 
def test_model():
    global w
    global b
    loss = 0.0
    correct = 0
    for i in range(10000):
        x = x_test[i].reshape(dim, 1)
        y = y_test[i].reshape(10, 1)
        z = w.T.dot(x) + b 
        a = softmax(z)
        loss += cce(y, z) 
        if np.argmax(a) == np.argmax(y):
            correct += 1

    print("Accuracy - %0.4f" % ((correct / 10000) * 100))
        
# Train the model.
train_model_mini_batch(20, 32, 0.01)
print("Model Trained!")
# Test the model. 
test_model()
