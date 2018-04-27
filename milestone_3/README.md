Milestone 3
===========

Code and Library
-------
Our team used 
-GPML library for MATLAB which train Gaussian Process Regression by minimize the negative log marginal likelihood. <br />
-MATLAB function fitlm to fit a linear regression to the training data and use function crossval to to 10 fold cross validation for the test data. <br />


Description
------------

Our original project use Bike Sharing Data, but there is category feature in the data, so we found another data to work on this milestone. <br/>
For this milestone, the data we are using is Fire Forest data from UCI Machine Learning Repository.  The data provide information on the meteorological data and try to predict forest fire. Available at: http://www.dsi.uminho.pt/~pcortez/fires.pdf <br/>
It is noted that several of the attributes in this dataset may be correlated, thus it makes sense to apply some sort of feature selection. Therefore, our goal is to apply PCA/SVD to this dataset and run Linear Regression and Gaussian Process Regression on the principal components, then compare the result with when we fit Linear Regression and Gaussian Process regression to the original dataset.

Method
--------
Use built-in function PCA in MATLAB to apply PCA to the data. <br/>
Use FITLM to fit the linear regression, <br/> 
Use GPML library for MATLAB to fit a gaussian process regression for our data. <br/>

Resources
------------
MATLAB
GMPL
FITLM
CROSSVAL

Result
------------
Apply PCA:

![alt text](https://raw.githubusercontent.com/sithuaung223/MachineLearning-ApplicationProject/master/milestone_3/proportionofvar.jpg)


Visualize the data:

![alt text](https://raw.githubusercontent.com/sithuaung223/MachineLearning-ApplicationProject/master/milestone_3/scatterredblue.jpg)

Linear Regression

1. For the orginal data <br/>
![alt text](https://raw.githubusercontent.com/sithuaung223/MachineLearning-ApplicationProject/master/milestone_3/lm1.PNG)

2. Principal Components <br/>
![alt text](https://raw.githubusercontent.com/sithuaung223/MachineLearning-ApplicationProject/master/milestone_3/lm2.PNG)


Gaussian Process Regression: <br/>
Running the gaussian process regression on the orginal data gives us negative log marginal likelihood to be -13.8155 <br/>

Running the gaussian process regression up to the 4th principal components gives us the negative log marginal likelihood of the model to be -13.7564 <br/>




Running the code
----------------
Using MATLAB to run PCA.m and GProcess.m


