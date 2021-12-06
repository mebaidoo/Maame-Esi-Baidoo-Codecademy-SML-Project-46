import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")
#Inspecting the dataframe
print(df.head())
#Calculating the mean of the total production of honey per year
prod_per_year = df.groupby("year").totalprod.mean().reset_index()
print(prod_per_year)
#Saving the different columns in prod_per_year in different variables
X = prod_per_year["year"]
X = X.values.reshape(-1, 1)
print(X)
y = prod_per_year["totalprod"]
print(y)
#Plotting a scatter plot of total production per year against year
plt.scatter(X, y)
plt.show()
#Creating and fitting a linear regression model
regr = linear_model.LinearRegression()
regr.fit(X, y)
#Getting the slope and intercept of the line
print(regr.coef_[0])
print(regr.intercept_)
#y_predict will contain the predictions the model would make on the X data
y_predict = regr.predict(X)
#print(y_predict)
#Plotting the y predicted values against X as a line on top of the scatter plot (line of best fit)
plt.plot(X, y_predict)
plt.show()
#Looks like the production of honey has been in decline over the years
#Using the model to predict what honey production may look like in the year 2050
#Creating a numpy array of years from 2013 to 2050
X_future = np.array(range(2013, 2051))
#Reshaping the array
X_future = X_future.reshape(-1, 1)
#print(X_future)
#future_predict will contain the y_values that the model would predict for the values of X_future, that is, 2013 to 2050
future_predict = regr.predict(X_future)
print(future_predict)
#Less and less honey production per year in the future
#Plotting future_predict against X_future
plt.clf()
plt.plot(X_future, future_predict)
plt.show()