import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import layers
from keras import models
from keras import optimizers
from keras.utils import normalize
from keras.datasets import mnist

# Create visited array for DFS
visited = [0] * 784

# Create Parent array for Disjoint Set Alg
parents = list(range(0, 784))

# Retrieve data set. 
(x_train_original, y_train), (x_test_original, y_test) = mnist.load_data()

count, rows, cols = x_train_original.shape

# Normalize Data Set. 
x_train = normalize(x_train_original, axis=1)
x_test = normalize(x_test_original, axis=1)
x_train = x_train.reshape(60000, rows * cols, 1)
x_test = x_test.reshape(10000, rows * cols, 1)

# Get data set for custom features 
x_train_bw = x_train_original
x_test_bw = x_test_original

# Replace all gray values with black. 
x_train_bw[x_train_bw > 0] = 1
x_test_bw[x_test_bw > 0] = 1

# Add new feature value to the end ot the input vector.
def add_features(train, test):
    global x_train_bw
    global x_test_bw
    for p in range(count):
        train[p] = np.append(train[p], disjoint_sets(x_train_bw[p]))

    for q in range(10000):
        test[p] = np.append(test[q], disjoint_sets(x_test_bw[q]))

# Get root of each set. This will act as path compression. 
def find_parent(v):
    n = v
    while n != parents[n]:
        n = parents[n]
    return n 

# Use DFS to search through all connected pixels and updating a 
# set list using path compression. 
def dfs(x, i, j):
    
    global visited
    global parents
    
    # Create search arrays to check adjacent pixels. 
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]

    # Mark this pixel visited. 
    current_pixel = (i * rows) + j
    visited[current_pixel] = 1

    # If we are on a black pixel mark it as visited and move on. 
    if x[i][j] == 1:
        parents[current_pixel] = -1
        return

    # Check adjacent pixels using dx, dy arrays. 
    for n in range(4):
        new_x = j + dx[n]
        new_y = i + dy[n]
        next_pixel = (i * rows) + j + (cols * dy[n]) + dx[n]

        # If a black pixel is found, mark that as negative on in the parents array. 
        if 0 <= new_y < rows and 0 <= new_x < cols:
            if x[new_y][new_x] == 1:
                parents[next_pixel] = -1

        # Check pixels below, to the right, above and left of current position. 
        # If pixel of value 0 is found and has not been visited, mark that pixel as having 
        # current pixel as its parent and move to that pixel using recursion.  
        if 0 <= new_y < rows and 0 <= new_x < cols:
            if x[new_y][new_x] == 0 and visited[next_pixel] == 0:
                parents[next_pixel] = find_parent(current_pixel)
                dfs(x, new_y, new_x)


def disjoint_sets(x):
    #
    # Algorithm:  
    #   - Start at (0, 0) and look down, right, up, left.
    #   - The first pixel found, mark current pixel as that new pixels parent.
    #   - Mark current pixel visited. 
    #   - Move to new pixel. 
    #   - Repeat until all nodes are visited 
    #
    global visited
    global parents

    for i in range(28):
        for j in range(28):
            current_pixel = (i * rows) + j
            if visited[current_pixel] == 0:
                dfs(x, i, j)

    # Return the number of white regions.  This should correspond to how many elements are 
    # in the parents array minus 1. 
    return len(np.unique(parents)) - 1 


add_features(x_train, x_test)
# Create Model
model = models.Sequential()
model.add(layers.Dense(128, input_shape=(rows * cols + 1, 1), activation='relu'))
model.add(layers.Dropout(rate=0.1))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='relu'))
sgd = optimizers.SGD(lr=0.01)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=32, verbose=1)

# Retrieve a list of accuracy results on training and test data
# sets for each training epoch
acc = history.history['acc']
val_acc = history.history['val_acc']

# Retrieve a list of list results on training and test data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')



