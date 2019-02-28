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
train_images = train_images_original.reshape(count, dim)
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
        return 1 / (1 + np.exp(-z))

    # Define Squared Error Loss
    def se(self, y, a):
        return 0.5 * ((a - y.T) ** 2)

    # Define Squared Error Loss Prime
    def se_prime_w(self, x, a, y, z):
        dz = a - y 
        da = (1 - a) * a
        dw = np.multiply(dz, da) 
        dw = x.T.dot(dw)
        return dw 

    def se_prime_b(self, a, y, z):
        db = (a - y.T) * (1 - a) * a
        return np.sum(db)

    def train_model_se(self, epochs, learning_rate, batch_size):
        
        for epoch in range(epochs):
            s = np.random.permutation(count)
            x_shuffled = self.tri[s]
            y_shuffled = self.trl[s]
            for i in range(0, count, batch_size):
                x = x_shuffled[i:i+batch_size]
                y = y_shuffled[i:i+batch_size]
                y = y[:,self.number].reshape(batch_size,1)
                w = self.weights
                b = self.bias
                z = (w.T.dot(x.T) + b).T
                a = self.sigmoid(z) 
                dw = self.se_prime_w(x, a, y, z) * (1 / batch_size)
                db = self.se_prime_b(a, y, z) * (1 / batch_size)
                self.weights = w - (dw * learning_rate)
                self.bias = b - (db * learning_rate)
                
        print("Classifier %d trained." % self.number)
           
    def make_prediction(self, ti):
        w = self.weights
        b = self.bias
        x = ti
        z = (w.T.dot(x.T) + b).T
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
epochs = 10
learning_rate = 0.5
batch_size = 32

model_zero.train_model_se(epochs, learning_rate, batch_size)
model_one.train_model_se(epochs, learning_rate, batch_size)
model_two.train_model_se(epochs, learning_rate, batch_size)
model_three.train_model_se(epochs, learning_rate, batch_size)
model_four.train_model_se(epochs, learning_rate, batch_size)
model_five.train_model_se(epochs, learning_rate, batch_size)
model_six.train_model_se(epochs, learning_rate, batch_size)
model_seven.train_model_se(epochs, learning_rate, batch_size)
model_eight.train_model_se(epochs, learning_rate, batch_size)
model_nine.train_model_se(epochs, learning_rate, batch_size)

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
