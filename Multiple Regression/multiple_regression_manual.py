import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time

def initialize_theta(x):
    # Random initialization for theta (one bias term and the rest for features)
    return np.random.rand(x.shape[1] + 1)

def predict(x, theta):
    return np.dot(x, theta)

def loss(y,predictions):
    return np.sum((y - predictions) ** 2)
def multiple_regression_delta_0_loss_ratio(x, y, number_of_iters=1000000, threshold=320, stepsize=0.000003):
    theta = initialize_theta(x)

    losses = []
    x_with_bias = np.c_[np.ones(x.shape[0]), x]
    n = len(y)
    column_squares = np.sum(x_with_bias**2, axis=0)
    start_time = time.time()
    while time.time() - start_time < threshold:
        predictions = predict(x_with_bias,theta)
        gradient = -2 * np.dot(x_with_bias.T, (y - predictions))
        gradient = gradient / column_squares
        loss_ratio = loss(y,predictions) / loss(y,predict(x_with_bias,-gradient))
        theta = theta * (1 /(1 + loss_ratio)) - gradient * stepsize * (loss_ratio / (loss_ratio + 1))
        losses.append(loss(y,predictions)//1)

    return theta,losses
def multiple_regression_delta_0(x, y, number_of_iters=1000000, threshold=320, stepsize=0.000003):
    theta = initialize_theta(x)

    losses = []
    x_with_bias = np.c_[np.ones(x.shape[0]), x]
    n = len(y)
    column_squares = np.sum(x_with_bias**2, axis=0)
    start_time = time.time()
    while time.time() - start_time < threshold:
        predictions = predict(x_with_bias,theta)
        gradient = -2 * np.dot(x_with_bias.T, (y - predictions))
        gradient = gradient / column_squares

        loss_ratio = loss(y,predictions) / loss(y,predict(x_with_bias,-gradient))
        if(np.sum(np.abs(theta * (1 - stepsize) - gradient * stepsize -theta))<0.000001):
            break
        theta = theta * (1 - stepsize) - gradient * stepsize
        losses.append(loss(y,predictions)//1)

    return theta,losses

def multiple_regression_gradient_descent(x, y, number_of_iters=1000000, threshold=120,stepsize=0.000003):
    # Initialize theta with random values
    theta = initialize_theta(x)

    losses = []
    x_with_bias = np.c_[np.ones(x.shape[0]), x]

    start_time = time.time()

    while time.time() - start_time < threshold:
        # Calculate predictions
        predictions = predict(x_with_bias, theta)

        # Calculate gradient
        gradient = -2 / len(y) * np.dot(x_with_bias.T, (y - predictions))

        # Update theta using gradient and step size
        theta_new = theta - stepsize * gradient

        # Exit on convergence
        if(np.sum(np.abs(theta_new - theta)) < 0.000001):
            break

        losses.append(loss(y,predictions)//1)
        theta = theta_new


    return theta,losses


df = pd.read_csv("../DataSets/Boston Housing/housing.data", delimiter=r"\s+")
column_names= ['Crime Rate', 'Residential Land Zoned', 'Non-retail Business Acres', 'Charles River Dummy', 'Nitric Oxides Concentration', 'Rooms per Dwelling', 'Owner Occupied pre1940', 'Weighted Distance to Employment Centres', 'Radial Highway Accessibility Index', 'Property Tax', 'Pupil Teacher Ratio', 'Blacks', 'Lower Status', 'Median Value']

df.columns = column_names

#print(df.head())


y = df[df.columns[-1]]

print(y)

x = df[df.columns[:-1]]
#x = np.array([[1,10,13],[2,4,-5]]).T
#y = np.array([5,18,78])

# Time the model training

start_time = time.time()

model,losses = multiple_regression_gradient_descent(x, y)

end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)
# Predict values and calculate loss
x_with_bias = np.c_[np.ones(x.shape[0]), x]  # Adding bias term to x for prediction
predictions = predict(x_with_bias, model)

with open("output.txt", "w") as file:
    file.write(f'Model Parameters: {model}\n')
    file.write(f'Loss: {loss}\n')
    file.write('Predictions:\n')
    for pred_i,pred in enumerate(predictions):
        file.write(f'{y[pred_i] - pred}\n')

print("Results written to output.txt")
plt.plot(losses[-100:], marker='o')
plt.xlabel('İterasyon')
plt.ylabel('Loss Değeri')
plt.title('Gradient Descent İterasyonlarındaki Loss Değerleri')
plt.grid(True)
plt.show()

# Time the model training
start_time = time.time()

model_delta_0,losses = multiple_regression_delta_0(x, y)

end_time = time.time()
elapsed_time = end_time - start_time
x_with_bias = np.c_[np.ones(x.shape[0]), x]  # Adding bias term to x for prediction
predictions = predict(x_with_bias, model_delta_0)
with open("output_delta_0.txt", "w") as file:
    file.write(f'Model Parameters: {model_delta_0}\n')
    file.write(f'Loss: {loss(y,predictions)}\n')
    file.write('Predictions:\n')
    for pred_i,pred in enumerate(predictions):
        file.write(f'{y[pred_i] - pred}\n')
print(elapsed_time)
print("Results written to output.txt")
plt.figure(figsize=(10, 6))
plt.plot(losses[-100:], marker='o')
plt.xlabel('İterasyon')
plt.ylabel('Loss Değeri')
plt.title('Gradient Descent İterasyonlarındaki Loss Değerleri')
plt.grid(True)
plt.show()