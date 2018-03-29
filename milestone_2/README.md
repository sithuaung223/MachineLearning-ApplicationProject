Milestone 1
===========

Code and Library
-------
Our team used 
-GPML library for MATLAB which train Gaussian Process Regression by minimize the negative log marginal likelihood


Description
------------

Our team chose the Bike Sharing data from UCI Machine Learning Repository as our dataset to use in application project. Bike-sharing system is a world wide growing program with 500,000 bicycle and also played an important role in traffic, environmental and health issue. Therefore, those data are useful for research that are related to sensing mobility in city. Also, it is expected that the most important event in city could be detected or predicted via monitoring these data.
Our goal is run Gaussian Process Regression on the data, which fit the data set by minimize the negative log marginal likelihood, using two different kernel: Squared Exponential and Matern 3/2 Kernel. Then we compare the two methods using the negative log marginal likelihood

Method
--------
We used GPML library for MATLAB to fit a gaussian process regression for our data

Resources
------------
MATLAB
GMPL

Result
------------
Using squared exponential kernel:
Negative log marginal likelihood:

![alt text](https://raw.githubusercontent.com/sithuaung223/MachineLearning-ApplicationProject/master/milestone_2/SquaredExponentialKernel.jpg)



Using Matern 3/2 kernel
Negative log marginal likelihood
![alt text](https://raw.githubusercontent.com/sithuaung223/MachineLearning-ApplicationProject/master/milestone_2/Matern32.jpg)


Running the code
----------------
Using MATLAB to run SquaredExponentialKernel.m to get the result using the squared exponential kernel and run Matern32.m to get the result using Matern 3/2 kernel


