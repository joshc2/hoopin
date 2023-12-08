
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import requests
import re
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


#ALL COLUMNS
#all_columns = ['FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', "MOV", "Pace", "TS%", 'W', 'L']

def select_columns(SC=['3P%','2P%', 'AST','TRB','STL',"TS%", 'W', 'L']):
    return SC

# COLUMNS WE ARE KEEPING
selected_columns = ['3P%','2P%', 'AST','TRB','STL',"TS%", 'W', 'L']


# # Split the data into predictor variables (X) and target variable (Y)
# X = all_data_1[selected_columns[:-2]]  # All columns except 'W' and 'L'
# Y = all_data_1['W']

# # Split the data into training and testing sets
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# # Create a linear regression model, fit it to the training data, and make predictions
# model = LinearRegression()
# model.fit(X_train, Y_train)
# Y_pred = model.predict(X_test)

# # Calculate the mean squared error (MSE) to evaluate the model's performance
# mse = mean_squared_error(Y_test, Y_pred)
# print("Mean Squared Error:", mse)

# # Inspect the coefficients of the linear regression model to determine variable importance
# coefficients = pd.Series(model.coef_, index=X.columns)
# sorted_coefficients = coefficients.abs().sort_values(ascending=False)
# print("Most important variables:")
# print(sorted_coefficients)






#ALL COLUMNS
#all_columns = ['FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', "MOV", "Pace", "TS%", 'W', 'L']

def select_columns(SC=['3P%','2P%', 'AST','TRB','STL',"TS%", 'W', 'L']):
    return SC

# COLUMNS WE ARE KEEPING
selected_columns = ['3P%','2P%', 'AST','TRB','STL',"TS%", 'W', 'L']


# Split the data into predictor variables (X) and target variable (Y)
def run_regression(selected_columns, all_data_1):
    X = all_data_1[selected_columns[:-2]]  # All columns except 'W' and 'L'
    Y = all_data_1['W']

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # Create a linear regression model, fit it to the training data, and make predictions
    model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    # Calculate the mean squared error (MSE) to evaluate the model's performance
    mse = mean_squared_error(Y_test, Y_pred)
    print("Mean Squared Error:", mse)

    # Inspect the coefficients of the linear regression model to determine variable importance
    coefficients = pd.Series(model.coef_, index=X.columns)
    sorted_coefficients = coefficients.abs().sort_values(ascending=False)
    print("\nMost important variables:")
    print(sorted_coefficients)

    return
