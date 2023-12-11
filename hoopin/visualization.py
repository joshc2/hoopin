import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import pkg_resources
import statsmodels

path = 'data/basketball.csv'

data_path = pkg_resources.resource_filename('hoopin', path)


all_data_1 =  pd.read_csv(data_path)

selected_columns = ['3P%','2P%', 'AST','TRB','STL',"TS%", 'W', 'L']


# Split the data into predictor variables (X) and target variable (Y)
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

# Inspect the coefficients of the linear regression model to determine variable importance
coefficients = pd.Series(model.coef_, index=X.columns)
sorted_coefficients = coefficients.abs().sort_values(ascending=False)


# Visualization 2: Bar chart of Coefficients
coefficients = coefficients.sort_values(ascending=True)
coefficients = pd.Series(model.coef_, index=X.columns)
coefficients = coefficients.sort_values(ascending=True)



def show_actual_predicted():
    """
    Visualizes the relationship between actual and predicted values of a target variable.

    This function creates a scatter plot to compare the actual values (Y_test) with the predicted values (Y_pred).
    It helps in visually assessing how well a predictive model aligns with the true outcomes.

    """   
    # Visualization 1: Scatter plot of Actual vs Predicted Wins
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=Y_test, y=Y_pred)
    plt.title('Actual vs Predicted Wins')
    plt.xlabel('Actual Wins')
    plt.ylabel('Predicted Wins')
    plt.show()
    return    

    # Additional Evaluation Metrics
def show_MSE():
    """
    Calculates and displays the Mean Squared Error (MSE) between actual and predicted values.

    The function computes the MSE, a metric that quantifies the average squared difference between actual
    and predicted values of a target variable. A lower MSE indicates better model performance.
    """ 
    mse = mean_squared_error(Y_test, Y_pred)
    print("Mean Squared Error:", mse)
    return
 
def show_r2():
    """
    Calculates and displays the R-squared (coefficient of determination) between actual and predicted values.

    The function computes the R-squared, a metric that quantifies the proportion of the variance in the
    dependent variable that is predictable from the independent variable. A higher R-squared indicates
    better explanatory power of the model.
    """

    r2 = r2_score(Y_test, Y_pred)
    print("R-squared:", r2)
    return

def show_residuals():
    """
    Visualizes the residuals (the differences between actual and predicted values) using a residual plot.

    The function creates a residual plot to help assess the goodness of fit of a predictive model. Residuals
    represent the vertical distances between data points and the regression line. A well-fitted model would
    have residuals randomly scattered around zero.
    """
    # RESIDUAL PLOT
    plt.figure(figsize=(10, 6))
    sns.residplot(x=Y_pred, y=Y_test, lowess=True, color="g")
    plt.title('Residual Plot')
    plt.xlabel('Predicted Wins')
    plt.ylabel('Residuals')
    plt.show()
    return


def show_residuals_distribution():
    """
    Visualizes the distribution of residuals (the differences between actual and predicted values).

    The function creates a distribution plot to show the spread and shape of residuals. Understanding the
    distribution of residuals can provide insights into the model's performance and identify any patterns
    in the errors.
    """

    # distribution plot of residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(Y_test - Y_pred, kde=True)
    plt.title('Distribution of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()
    return

def show_pairplot():
    """
    Visualizes pairwise relationships between selected columns in a dataset using a pair plot.

    The function creates a pair plot to display scatterplots for each pair of selected columns, histograms
    along the diagonal, and additional information like kernel density estimates.

    """
    #Pair PLOT
    sns.pairplot(all_data_1[selected_columns])
    plt.suptitle('Pair Plot of Selected Columns', y=1.02)
    plt.show()
    return

def show_correlation_heatmap():
    """
    Visualizes the correlation between selected columns in a dataset using a heatmap.

    The function creates a correlation heatmap to illustrate the strength and direction of the linear
    relationship between pairs of selected columns. Correlation values are annotated on the heatmap.

    """
    # CORRELATION HEAT MAP
    plt.figure(figsize=(10, 8))
    sns.heatmap(all_data_1[selected_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Selected Columns')
    plt.show()
    return
    





    # path = 'data/basketball.csv'

    # data_path = pkg_resources.resource_filename('hoopin', path)


    # all_data_1 =  pd.read_csv(data_path)

    # selected_columns = ['3P%','2P%', 'AST','TRB','STL',"TS%", 'W', 'L']


    # # Split the data into predictor variables (X) and target variable (Y)
    # X = all_data_1[selected_columns[:-2]]  # All columns except 'W' and 'L'
    # Y = all_data_1['W']

    # # Split the data into training and testing sets
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # # Create a linear regression model, fit it to the training data, and make predictions
    # model = LinearRegression()
    # model.fit(X_train, Y_train)
    # Y_pred = model.predict(X_test)



    # coefficients = pd.Series(model.coef_, index=X.columns)
    # sorted_coefficients = coefficients.abs().sort_values(ascending=False)

    ######




    
# #ALL COLUMNS
# #all_columns = ['FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', "MOV", "Pace", "TS%", 'W', 'L']

# # COLUMNS WE ARE KEEPING
# selected_columns = ['3P%','2P%', 'AST','TRB','STL',"TS%", 'W', 'L']


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

