# if __name__ == 'main':


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

# COLUMNS WE ARE KEEPING
selected_columns = ['3P%','2P%', 'AST','TRB','STL',"TS%", 'W', 'L']


    # Split the data into predictor variables (X) and target variable (Y)  all_data_1
def run_regression( all_data_1,selected_columns=['3P%','2P%', 'AST','TRB','STL',"TS%", 'W', 'L']):
    """
    Perform linear regression analysis on a dataset to predict the values in the 'W' column.

    Parameters
    ----------
    all_data_1 : pandas.DataFrame
        The input dataset.
    selected_columns : list
        List of column names to be used as independent variables in the regression.

    Returns
    -------
    mse : float
        Mean Squared Error (MSE) to evaluate the model's performance.
    sorted_coefficients : pandas.Series
        Coefficients of the linear regression model sorted by their absolute values,
        indicating the importance of each variable in predicting the target.

    Examples
    --------
    >>> run_regression(df, ['columns', 'dataset'])
    Mean Squared Error: 0.12345

    Most important variables:
    X2    0.56789
    X1    0.45678
    X3    0.23456
    """

    X = all_data_1[selected_columns]  # All columns except 'W' and 'L'
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




