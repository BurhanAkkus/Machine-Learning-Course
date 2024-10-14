import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

def initialize_theta(x):
    initial_theta = [random.random(),random.random()]
    if(len(x.shape) != 1):
        for i in range(x.shape[1] - 1):
            initial_theta.append(random.random())
    return initial_theta


def linear_regression(x, y, number_of_iters=5000000, threshold=0.000001):
    theta = initialize_theta(x)
    n = len(y)
    y_times_x1 = np.sum(x*y)
    x_sum = np.sum(x)
    y_sum = np.sum(y)
    x_squared_sum = np.sum(x**2)
    for iter in range(number_of_iters):
        theta_old = copy.deepcopy(theta)
        theta[0] = y_sum/n - theta_old[1]/n * np.sum(x)
        theta[1] = (y_times_x1 - theta_old[0]*x_sum) / x_squared_sum
        if(np.sum(np.subtract(theta,theta_old)**2) < threshold):
            break
    return theta



df = pd.read_csv("../DataSets/Boston Housing/housing.data", delimiter=r"\s+")
column_names= ['Crime Rate', 'Residential Land Zoned', 'Non-retail Business Acres', 'Charles River Dummy', 'Nitric Oxides Concentration', 'Rooms per Dwelling', 'Owner Occupied pre1940', 'Weighted Distance to Employment Centres', 'Radial Highway Accessibility Index', 'Property Tax', 'Pupil Teacher Ratio', 'Blacks', 'Lower Status', 'Median Value']

df.columns = column_names

#print(df.head())


y = df[df.columns[-1]]

print(y)

x = df['Lower Status']

model = linear_regression(x,y)

print(model)