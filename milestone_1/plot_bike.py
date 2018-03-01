#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv, sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the bike dataset
bike_X = []
bike_Y = []
with open('day.csv') as csvDataFile:
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
#print("test_dataSize", test_dataSize)
#print("X", bike_X[0])
#print("Y", bike_Y[0])
#sys.exit(0)

# Split the data into training/testing sets
bike_X_train = bike_X[:-test_dataSize]
bike_X_test = bike_X[-test_dataSize:]
#np.reshape(bike_X_train, -1, bike_X_train.shape[0])
#np.reshape(bike_X_test, -1, bike_X_test.shape[0]) 
print("X_train", bike_X_train.shape)
print("X_test", bike_X_test.shape)
#sys.exit(0)


# Split the targets into training/testing sets
bike_Y_train = bike_Y[:-test_dataSize]
bike_Y_test = bike_Y[-test_dataSize:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(bike_X_train, bike_Y_train)

# Make predictions using the testing set
bike_Y_pred = regr.predict(bike_X_test)

print("X_train", bike_X_train.shape)
print("X_test", bike_X_test.shape)
print("Y_train", bike_Y_train.shape)
print("Y_test", bike_Y_test.shape)
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
