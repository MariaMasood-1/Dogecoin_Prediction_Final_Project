# Dogecoin Prediction Final Project

## Summary: 
This project uses a machine learning model to explore the variability of dogecoin price in real time, construct a prediction curve and compare the mapping of prediction with actual data to understand the implementation of machine learning model along with learning multiple libraries usage in the modelling. This project will touch on the history and application of Dogecoin establishing the context of the project and further explain the libraries used and their core functions. Also, we'll explain how the machine learning model works and towards the end discuss the conclusion drawn out of the final graph.


## Dogecoin: 
Dogecoin is a cryptocurrency created by software engineers Billy Markus and Jackson Palmer, who decided to create a payment system as a joke, making fun of the wild speculation in cryptocurrencies at the time. Dogecoin is an open source peer to peer digital currency. It can be easily transferred from the internet throughout the world (wherever cryptocurrency is not banned). Also, dogecoin is accepted by multiple retailers. For example, if someone bought dogecoin at the rate of $0.002/dogecoin, and invested $20 to get 1000 dogecoins. Irrespective of the price variation of the cryptocurrency the buyer would have the value of 1000 dogecoins to trade for its worth. Dogecoin, a cryptocurrency that was created as a joke, has risen in price by more than 12,000% and hit a record 69 cents per token this week. But it is a highly volatile investment and involves a lot of risk. Even the price prediction of Dogecoin is not the usual stock market prediction where the traders can make a safe bet. 

### Price Prediction Model:
Predicting the price of a cryptocurrency is a regression problem in machine learning. We will be studying the historical prices of Dogecoin cryptocurrency employing the yahoo finance database to fetch the historical prices of Dogecoin. Dogecoin is very cheap right now, but financial experts are predicting that we may see a major increase in dogecoin prices.
 
There are multiple ways to construct a machine learning model that we can use for the task of Dogecoin price prediction. We use a simple machine learning model using python/object oriented programming using multiple libraries. We will dive deeper into the project now.
 
### Importing / Installing the packages
Before we start, Let’s install and import the required libraries such as numpy, pandas, matplotlib and SciKit-Learn as well as Yahoo’s finance packages, I’ve pip installed yahoo finance package since it was not available on google colab. We install these packages by doing this:

```
Python
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
<br>
-Numpy
<br>
-Pandas 
<br>
-matplotlib
<br>
-SciKit-learn
<br>
-Yfinance

### Downloading and refining the data:

The next step is to import the dataset using the yahoo finance finance package by implementing the yf. Command. yfinance aims to offer a reliable, threaded, and Pythonic way to download historical market data from Yahoo! finance.This will allow us to download the data using the ticker symbol and date range, this is the following code to do so:
```
Python
df = yf.download('DOGE-USD', start= '2021-01-01', end= '2021-04-01', progress=False)
```
 
We can print this data out using the df.head(n) function. This function returns the first n rows for the object based on position. It is useful for quickly testing if your object has the right type of data in it, whereas if no value of n is given then it’ll take n=5 as default.:

```
Python
 df = yf.download('DOGE-USD', start= '2021-01-01', end= '2021-04-01', progress=False)
 ```
Next up, we want to get the close price of our crypto and store it in a different variable, we also want to reshape that data frame, by reshaping we can add or remove dimensions or change the number of elements in each dimension. We convert the close to a one-dimensional array. We do so by using the following line: <br>
 ```
 Python
series = df['Close'].values.reshape(-1,1)
 ```
 
The reason we store this in another variable is because we’re going to fit our machine learning model to that specific value. Next up we have to normalize the data, it also often refers to rescaling by the minimum and range of the vector, to make all the elements lie between 0 and 1 thus bringing all the values of numeric columns in the dataset to a common scale. We first start off by declaring our scaler, this will allow us to have a mean value of 0 while having our standard deviation of 1, we would then fit our close data we created in the code above to the scaler we just created, we then declare the “series” variable back to the transformed scaler which is transformed into a 1D array using the “.flatten” command within Numpy. <br>
```

scaler = StandardScaler()      #creating a scalar with 0 mean and 1 standard deviation
scaler.fit(series[:len(series) // 2])      #fit close data to the scalar
series = scaler.transform(series).flatten() 
```
Now we must create some new data frames that will help us hold the data for us, we’ll be creating these variables / empty dataframes: <br>
```
T= 10
S =1
X= []
Y = []
```
We use the “T” variable created above that will define the days we will be including in order to predict the future. Next up, we’re going to use a for loop to go through our series data, let’s declare a for loop, we are going to use the following line: <br>
```
for t in range(len(series) - T):
 
```
Notice that we are using a lowercase “t” which is our counter in this specific example, next up let’s fill our for loop up, so now we want to store our series data using our counter into another variable (x in this example) by slicing the dataset, then append that data to our uppercase X data frame that we declared above and then we do the same thing but instead of slicing the dataset we’re going to be just using it as a counter within the series dataset, we then append that same data to the Y data frame we created earlier. here are those lines in our for loop: <br>
```
 x = series[t:t+T]
  X.append(x)
  y = series[t+T]
  Y.append(y)
```


Finally, we want to reshape our data frame, this will basically give a new shape to our data frame without changing any of the data in this data frame, we will then create an array for the “Y” data frame as well, finally we will get the length of the “X” array and store it in a new variable called “N”. <br>
```
X = np.array(X).reshape(-1, T)
Y = np.array(Y)
N = len(X)
print(“X.shape”, X.shape, “Y.shape”, Y.shape)
```
 
Awesome! We’re now going to have to create a class for our Machine Learning model, this is the fun stuff! Let’s start off by creating a class called BaselineModel, then define a function with the following code: <br>
```
#creating a class for our Machine Learning model
class BaselineModel:
  def predict(self, X):
    return X[:, -1] #return the last value for each input sequence
```

Next up we’re going to have to split up our data to a train and test set. We do so by creating the Xtrain & Train variables, we then use the “X” and “N” variables we used before to fill those variables with data, we essentially do the same thing with our “Xtest” and “Ytest” variables with the other half of the data for our test set: <br>
```


#split the data to test data and train data
Xtrain, Ytrain = X[:-N//2], Y[:-N//2]
Xtest, Ytest = X[-N//2:], Y[-N//2:]
```
Awesome! Next up let’s go ahead and setup our model, we’re going to create a “model” variable that holds our “BaselineModel” class, we’re going to create some new variables to pass our train and testing data frames, we do so by using the following code:   <br>
```

model = BaselineModel()
Ptrain = model.predict(Xtrain)
Ptest = model.predict(Xtest)
```
Great! Now we’re going to go ahead and reshape our arrays once more and store them into another variable as well as create the 1D array with Numpy: <br>
```

#reshaping the arrays and storing them into another variable and creating a 1D array with Numpy

Ytrain2 = scaler.inverse_transform(Ytrain.reshape(-1,1)).flatten()
Ytest2 = scaler.inverse_transform(Ytest.reshape(-1,1)).flatten()
Ptrain2 = scaler.inverse_transform(Ptrain.reshape(-1,1)).flatten()
Ptest2 = scaler.inverse_transform(Ptest.reshape(-1,1)).flatten()
```
Almost Done! Now we’re going to go ahead and send our data to pretty much be forecasted, the future data will be appended into our “forecast” variable, then our data will be plotted using the package matplotlib! This is the code to do that: <br>
```
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

 
 
