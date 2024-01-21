import pandas as pd
import numpy as np
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

# Read our data from csv file
data = pd.read_csv('data.csv', header=None)
data.head()

# Create dataset from Pandas dataframe
X = data.loc[:,0:1]
y = data.loc[:,2]

# Convert dataset to Numpy
X = X.to_numpy()
y = y.to_numpy()


# Helper functions to make a prediction
def stepFunction(t):
    if t >= 0:
        return 1
    return 0


def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])


# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_h = prediction(X[i], W, b)
        if y_h != y[i]:
            if y_h > 0:
                W[0] -= X[i][0] * learn_rate
                W[1] -= X[i][1] * learn_rate
                b -= learn_rate
            else:
                W[0] += X[i][0] * learn_rate
                W[1] += X[i][1] * learn_rate
                b += learn_rate

    return W, b


# This function runs the perceptron algorithm repeatedly on the dataset,
def trainPerceptronAlgorithm(X, y, learn_rate=0.01, num_epochs=25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])

    W = np.array(np.random.rand(2, 1))
    b = np.random.rand(1)[0] + x_max

    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step to update W & b
        W, b = perceptronStep(X, y, W, b, learn_rate)

        print(f"Training epoch: #{i + 1}")
        print(f"Weight: {W}")
        print(f"bias: {b}")

    return W, b

# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
model_train = trainPerceptronAlgorithm(X, y,
                                 learn_rate=0.001,
                                 num_epochs=10)


# Make a prediction from the model training
final_model = model_train
weight, bias = final_model

print(f"weight: {weight}")
print(f"bias: {bias}")

print(prediction(X=[-1, 0.5], W=weight, b=bias))