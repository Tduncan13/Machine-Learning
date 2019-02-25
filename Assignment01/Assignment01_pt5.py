import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import random
import sys

# sys.setrecursionlimit(1500)
# Create visited array for DFS
visited = [0] * 784

# Create Parent array for Disjoint Set Alg
parents = list(range(0, 784))

# Retrieve data set. 
mnist = tf.keras.datasets.mnist 
(x_train_original, y_train), (x_test_original, y_test) = mnist.load_data()

count, rows, cols = x_train_original.shape

# Normalize Data Set. 
x_train = tf.keras.utils.normalize(x_train_original, axis=1)
x_test = tf.keras.utils.normalize(x_test_original, axis=1)

# Get data set for custom features 
x_train_bw = x_train_original
x_test_bw = x_test_original

# Replace all gray values with black. 
x_train_bw[x_train_bw > 0] = 1
x_test_bw[x_test_bw > 0] = 1

# Get root of each set. This will act as path compression. 
def find_parent(v):
    n = v
    while n != parents[n]:
        n = parents[n]

    return n 

# Use DFS to search through all connected pixels and updating a 
# Set list using path compression. 
def dfs(x, i, j):
    
    global visited
    global parents
    
    # Create search arrays. 
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]

    # Mark this pixel visited. 
    current_pixel = (i * rows) + j
    visited[current_pixel] = 1
    if x[i][j] == 1:
        parents[current_pixel] = -1
        return
    # print(visited)
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
    # Start at (0, 0) and look down, right, up, left.
    # First pixel found, place current pixel location as that pixels parent.
    # Mark current pixel visited. 
    # Move to new pixel. 
    # Repeat until all nodes are visited 
    #
    global visited
    global parents

    regions = []

    for i in range(28):
        for j in range(28):
            current_pixel = (i * rows) + j
            if visited[current_pixel] == 0:
                dfs(x, i, j)

    print(len(np.unique(parents)) - 1)
    # for i in range(784):
    #     if parents[i] != -1:
            


# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

# model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=10, batch_size=32)

# test_loss, test_acc = model.evaluate(x_test, y_test)
# print(test_loss, test_acc)

test_labels = to_categorical(y_test)
q = 3
print(np.argmax(test_labels[q]))
disjoint_sets(x_test_bw[q])
# plt.imshow(x_test_bw[q])
# plt.show()