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
# 80% train data and 20% test data 
test_dataSize = int(len(bike_X) * 0.2)

# Split the data into training/testing sets
bike_X_train = bike_X[:-test_dataSize]
bike_X_test = bike_X[-test_dataSize:]


# Split the targets into training/testing sets
bike_Y_train = bike_Y[:-test_dataSize]
bike_Y_test = bike_Y[-test_dataSize:]

# Create linear regression object
regr = linear_model.LinearRegression()

sum_MSE = 0
kf = KFold(n_splits = 10, shuffle = True, random_state=0)
for train, test in kf.split(bike_X_train):
    # Split the data into training/testing sets
    X_CVtrain, X_CVtest = bike_X_train[train] , bike_X_train[test]

    # Split the targets into training/testing sets
    Y_CVtrain, Y_CVtest = bike_Y_train[train] , bike_Y_train[test]

    # Train the model using the training sets
    regr.fit(X_CVtrain, Y_CVtrain)

    # Make predictions using the testing set
    Y_CVpred = regr.predict(X_CVtest)
    
    sum_MSE += mean_squared_error(Y_CVtest, Y_CVpred)

avg = sum_MSE/10.00
print("Avearge Mean squared error: %2.f" % avg)

# Make predictions using the testing set
bike_Y_pred = regr.predict(bike_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(bike_Y_test, bike_Y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(bike_Y_test, bike_Y_pred))

# Plot outputs
plt.scatter(bike_X_test[:,0], bike_Y_test[:,2],  color='black')
plt.plot(bike_X_test[:,0], bike_Y_pred[:,2], color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
