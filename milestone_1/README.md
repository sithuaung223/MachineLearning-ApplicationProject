Milestone 1
===========

Code and Library
-------
plot_bike.py
Our team used 
-sklearn library for linear regression and 10 fold cross validation to built the linear model and error measure
-matplotlib to plot the result of model.
-numpy to set up the data set.
-anaconda and ipython for GUI plot in window10

Description
------------

Our team chose the Bike Sharing data from UCI Machine Learning Repository as our dataset to use in application project. Bike-sharing system is a world wide growing program with 500,000 bicycle and also played an important role in traffic, environmental and health issue. Therefore, those data are useful for research that are related to sensing mobility in city. Also, it is expected that the most important event in city could be detected or predicted via monitoring these data.
Our goal is to build a linear model to predict the number of bike shared per day according to weather condition.

Method
--------
We used linear regression to train the data and build the model. Crossvalidation method to validate how well our model predict the data. Frist we divided the data into 80% traning data and 20% test data. Then we used 10 fold Crossvalidation method to split and train the Training data with linear regression model. We used mean-squared-error to measure our model prediction. After we trained with training data set, we used our linear model on test data and compare with the actual result and plot the those on test data.

Resources
------------
Anaconda,
Ipython,
Scikit learn lib,
Numpy,
matplotlib

Result
------------
Our average mean-squared-error is quite high in numerical value and high variance. That means our model is underfit. It should be underfit because we are using the 5 types of weather features such as weather situation, temperature, feeling temperature, humidity, and windspeed. If we get more environmental condition feature, it might increase our model. For high variance, we can reslove the issue by training the model with more data. If we used hourly dataset which has more data, the prediciton seems less variance than daily dataset. 

Running the code
----------------
Since we try to plot our data, we will need "anaconda" and "ipython" to plot the data for GUI plotting tool.
Then go to anaconda command line window, go to location of the file "plot_bike"
type : ipython
>>> run plot_bike.py

Outputs 
----------
Average MSE of 10-fold-CV:
MSE of test data:
Variance score:
Plot
