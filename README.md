# Dogecoin Prediction Final Project

## Summary: 
This project uses a machine learning model to explore the variability of dogecoin price in real time, construct a prediction curve and compare the mapping of prediction with actual data to understand the implementation of machine learning model along with learning multiple libraries usage in the modelling. This project will touch on the history and application of Dogecoin establishing the context of the project and further explain the libraries used and their core functions. Also, we'll explain how the machine learning model works and towards the end discuss the conclusion drawn out of the final graph.

## Goal of this project:
This project gives an overview on the price prediction strategies used in the financial analysis of stock market prediction and the similar modelling technique is used to predict the varying price of Dogecoin. In this project we'll learn:

* To implement a price prediction model
* To understand the use of multiple libraries such as numpy, pandas, matplotlib, etc
* The concept of statistical analysis such as reshaping the dataframe, normalization of the dataframe, etc
* How to use training and test data and the significance of using these datasets
* ploting of the graphs using matplotlib

## What do we need:
We only need the access to google colab notebook or any python idle to run the code and generate the reults

## Introduction:

### What is Dogecoin?
Dogecoin is a cryptocurrency created by software engineers Billy Markus and Jackson Palmer, who decided to create a payment system as a joke, making fun of the wild speculation in cryptocurrencies at the time. Dogecoin is an open source peer to peer digital currency. It can be easily transferred from the internet throughout the world (wherever cryptocurrency is not banned). Also, dogecoin is accepted by multiple retailers. For example, if someone bought dogecoin at the rate of $0.002/dogecoin, and invested $20 to get 1000 dogecoins. Irrespective of the price variation of the cryptocurrency the buyer would have the value of 1000 dogecoins to trade for its worth. Dogecoin, a cryptocurrency that was created as a joke, has risen in price by more than 12,000% and hit a record 69 cents per token this week. But it is a highly volatile investment and involves a lot of risk. Even the price prediction of Dogecoin is not the usual stock market prediction where the traders can make a safe bet. 

### What is Price Prediction Model:
Predicting the price of a cryptocurrency is a regression problem in machine learning. We will be studying the historical prices of Dogecoin cryptocurrency employing the yahoo finance database to fetch the historical prices of Dogecoin. Dogecoin is very cheap right now, but financial experts are predicting that we may see a major increase in dogecoin prices.
 
There are multiple ways to construct a machine learning model that we can use for the task of Dogecoin price prediction. We use a simple machine learning model using python/object oriented programming using multiple libraries. We will dive deeper into the project now.

## Implementation of the model-
 
### Importing / Installing the packages
Before we start, Let’s install and import the required libraries such as numpy, pandas, matplotlib and SciKit-Learn as well as Yahoo’s finance packages, I’ve pip installed yahoo finance package since it was not available on google colab. We install these packages by doing this:

```Python
!pip install yfinance
!pip install yahoofinancials!pip install yfinance
!pip install yahoofinancials
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae 
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from yahoofinancials import YahooFinancials
```

### Function of different libraries

* Numpy: Numpy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.<br>

* Pandas: Pandas is defined as an open-source library that provides high-performance data manipulation in Python. It is built on top of the NumPy package, which means Numpy is required for operating the Pandas. The name of Pandas is derived from the word Panel Data, which means an Econometrics from Multidimensional data. It is used for data analysis in Python and developed by Wes McKinney in 2008.

* Before Pandas, Python was capable for data preparation, but it only provided limited support for data analysis. So, Pandas came into the picture and enhanced the capabilities of data analysis. It can perform five significant steps required for processing and analysis of data irrespective of the origin of the data, i.e., load, manipulate, prepare, model, and analyze. The Pandas module mainly works with the tabular data, whereas the NumPy module works with the numerical data. So, together Numpy and Pandas provide a great accessibility to statistical analysis.<br>

* matplotlib: Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits like Tkinter, wxPython, Qt, or GTK.<br>

* SciKit-learn: SciKit-learn is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.<br>

* Yfinance: yfinance is a popular open source library developed by Ran Aroussi as a means to access the financial data available on Yahoo Finance. Yahoo Finance offers an excellent range of market data on stocks, bonds, currencies and cryptocurrencies. It also offers market news, reports and analysis and additionally options and fundamentals data- setting it apart from some of it’s competitors.

### Downloading and refining the data:

The next step is to import the dataset using the yahoo finance finance package by implementing the yf. Command. yfinance aims to offer a reliable, threaded, and Pythonic way to download historical market data from Yahoo! finance.This will allow us to download the data using the ticker symbol and date range, this is the following code to do so:
```Python
df = yf.download('DOGE-USD', start= '2021-01-01', end= '2021-04-01', progress=False)
```
 
We can print this data out using the df.head(n) function. This function returns the first n rows for the object based on position. It is useful for quickly testing if your object has the right type of data in it, whereas if no value of n is given then it’ll take n=5 as default.:

```Python

df.head()
 ```
 <img width="299" alt="computer programming for gis_final project_df head" src="https://user-images.githubusercontent.com/76536418/120230159-b9fa6600-c21c-11eb-97f4-0a3d389dbc12.png">

Next up, we want to get the close price of our crypto and store it in a different variable, we also want to reshape that data frame, by reshaping we can add or remove dimensions or change the number of elements in each dimension. We convert the close to a one-dimensional array. We do so by using the following line: <br>
 ```Python

series = df['Close'].values.reshape(-1,1)
 ```
 
The reason we store this in another variable is because we’re going to fit our machine learning model to that specific value. Next up we have to normalize the data, it also often refers to rescaling by the minimum and range of the vector, to make all the elements lie between 0 and 1 thus bringing all the values of numeric columns in the dataset to a common scale. We first start off by declaring our scaler, this will allow us to have a mean value of 0 while having our standard deviation of 1, we would then fit our close data we created in the code above to the scaler we just created, we then declare the “series” variable back to the transformed scaler which is transformed into a 1D array using the “.flatten” command within Numpy. <br>
```Python;

scaler = StandardScaler()      #creating a scalar with 0 mean and 1 standard deviation
scaler.fit(series[:len(series) // 2])      #fit close data to the scalar
series = scaler.transform(series).flatten() 
```
Now we must create some new data frames that will help us hold the data for us, we’ll be creating these variables / empty dataframes: <br>
```Python
T= 10
S =1
X= []
Y = []
```
We use the “T” variable created above that will define the days we will be including in order to predict the future. Next up, we’re going to use a for loop to go through our series data, let’s declare a for loop, we are going to use the following line: <br>
```Python
for t in range(len(series) - T):
 
```
Notice that we are using a lowercase “t” which is our counter in this specific example, next up let’s fill our for loop up, so now we want to store our series data using our counter into another variable (x in this example) by slicing the dataset, then append that data to our uppercase X data frame that we declared above and then we do the same thing but instead of slicing the dataset we’re going to be just using it as a counter within the series dataset, we then append that same data to the Y data frame we created earlier. here are those lines in our for loop: <br>
```Python
 x = series[t:t+T]
  X.append(x)
  y = series[t+T]
  Y.append(y)
```
### Creating the Machine Learning Model:


Finally, we want to reshape our data frame, this will basically give a new shape to our data frame without changing any of the data in this data frame, we will then create an array for the “Y” data frame as well, finally we will get the length of the “X” array and store it in a new variable called “N”. <br>
```Python
X = np.array(X).reshape(-1, T)
Y = np.array(Y)
N = len(X)
print(“X.shape”, X.shape, “Y.shape”, Y.shape)
```
 
We’re now going to have to create a class for our Machine Learning model. Let’s start off by creating a class called BaselineModel, then define a function with the following code: <br>
```Python
#creating a class for our Machine Learning model
class BaselineModel:
  def predict(self, X):
    return X[:, -1] #return the last value for each input sequence
```

Next up we’re going to have to split up our data to a train and test set. We do so by creating the Xtrain & Train variables,The model is initially fit on a training dataset,[3] which is a set of examples used to fit the parameters (e.g. weights of connections between neurons in artificial neural networks) of the model. Successively, the fitted model is used to predict the responses for the observations in a second dataset called the validation dataset. Finally, the test dataset is a dataset used to provide an unbiased evaluation of a final model fit on the training dataset.. However, we're not creating a validation dataset in this analysis for the simplicity of the model. We have used the “X” and “N” variables we used before to fill those variables with data, we essentially do the same thing with our “Xtest” and “Ytest” variables with the other half of the data for our test set: <br>
```Python


#split the data to test data and train data
Xtrain, Ytrain = X[:-N//2], Y[:-N//2]
Xtest, Ytest = X[-N//2:], Y[-N//2:]
```
We now setup our model, we’re going to create a “model” variable that holds our “BaselineModel” class, we’re going to create some new variables to pass our train and testing data frames, we do so by using the following code:   <br>
```Python

model = BaselineModel()
Ptrain = model.predict(Xtrain)
Ptest = model.predict(Xtest)
```
Now we’re going to go ahead and reshape our arrays once more and store them into another variable as well as create the 1D array with Numpy: <br>
```Python

#reshaping the arrays and storing them into another variable and creating a 1D array with Numpy

Ytrain2 = scaler.inverse_transform(Ytrain.reshape(-1,1)).flatten()
Ytest2 = scaler.inverse_transform(Ytest.reshape(-1,1)).flatten()
Ptrain2 = scaler.inverse_transform(Ptrain.reshape(-1,1)).flatten()
Ptest2 = scaler.inverse_transform(Ptest.reshape(-1,1)).flatten()
```
### Forecasting of the Dogecoin Price:

Almost Done! Now we’re going to go ahead and send our data to pretty much be forecasted, the future data will be appended into our “forecast” variable, then our data will be plotted using the package matplotlib! This is the code to do that: <br>
```Python
#Forecasting the prediction data

forecast = []
input_ = Xtest[0]
while len(forecast) < len(Ytest):
  f = model.predict(input_.reshape(1,T))[0]
  forecast.append(f)
  #make a new input with the latest forecast
  input = np.roll(input_, -1)
  input_[-1] = f
plt.plot(Ytest, label = 'target')
plt.plot(forecast, label = 'prediction')
plt.legend()
plt.title("Right forecast")
plt.show()
```

And this is our output!

![dogecoin_forecast](https://user-images.githubusercontent.com/76536418/120227347-f925b880-c216-11eb-9bc0-b862a88e1c29.png)

## Challenges:
As we can see in the graph produced, the dogecoin price is highly volatile and is a huge risk for investment. This however is only a practice to learn for academic reason and not for real time investment decision. We may be able to modify and improve the modelling technique by including a validation set and increasing the number of features in order to improve the model performance metrics such as accuracy, recall and precision. Feature variable plays an important role in creating predictive models whether it is Regression or Classification Model. Having a large number of features is not good because it may lead to overfitting, which will make our model specifically fit the data on which it is trained. Also having a large number of features will cause the curse of dimensionality i.e. the features will increase the dimensions of search space for the problem.



## Resources: 
https://www.kaggle.com/tarandeep97/dogecoin-analysis

https://www.kaggle.com/kaushiksuresh147/doge-coin-to-moon-eda-and-prediction

https://lazyprogrammer.me/

https://preettheman.medium.com/how-to-predict-doge-coin-price-using-machine-learning-and-python-4bc7d723a6d3

https://www.wsj.com/articles/how-dogecoin-is-creating-a-frenzy-for-the-next-big-cryptocurrencyand-why-experts-advise-caution-11620411623


