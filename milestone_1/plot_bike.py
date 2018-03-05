#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv, sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

# Load the bike dataset
bike_X = []
bike_Y = []
with open('../dataset/day.csv') as csvDataFile:
    next(csvDataFile)
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        row.pop(1)
        r = np.array(row)
        Row = r.astype(float)
        bike_X.append(Row[:-3])
        bike_Y.append(Row[-3:])

bike_X = np.array(bike_X)
bike_Y = np.array(bike_Y)
# Create linear regression object
regr = linear_model.LinearRegression()

sum_MSE = 0.00
sum_variance = 0.00
kf = KFold(n_splits = 10, shuffle = True, random_state=0)
for train, test in kf.split(bike_X):
    # Split the data into training/testing sets
    X_CVtrain, X_CVtest = bike_X[train] , bike_X[test]

    # Split the targets into training/testing sets
    Y_CVtrain, Y_CVtest = bike_Y[train] , bike_Y[test]

    # Train the model using the training sets
    regr.fit(X_CVtrain, Y_CVtrain)

    # Make predictions using the testing set
    Y_CVpred = regr.predict(X_CVtest)
    
    sum_MSE += mean_squared_error(Y_CVtest[:,2], Y_CVpred[:,2])

    sum_variance += r2_score(Y_CVtest[:,2], Y_CVpred[:,2])


    # Plot outputs
    plt.scatter(X_CVtest[:,0], Y_CVtest[:,2],  color='black')
    plt.plot(X_CVtest[:,0], Y_CVpred[:,2], color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()

avg_MSE = sum_MSE/10.00
print("Average MSE of 10-fold-CV: %2.4f" % avg_MSE)

avg_variance = sum_variance/10.00
print("Average variance score of 10-fold-CV: %2.4f" % avg_variance)
