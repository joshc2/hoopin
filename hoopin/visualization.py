import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import pkg_resources
import statsmodels

# import hoopin
# Assuming you have already loaded your data and defined the variables as mentioned in your previous code

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
print("Mean Squared Error:", mse)

# Inspect the coefficients of the linear regression model to determine variable importance
coefficients = pd.Series(model.coef_, index=X.columns)
sorted_coefficients = coefficients.abs().sort_values(ascending=False)
print("Most important variables:")
print(sorted_coefficients)



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd






# Visualization 2: Bar chart of Coefficients
coefficients = coefficients.sort_values(ascending=True)
coefficients = pd.Series(model.coef_, index=X.columns)
coefficients = coefficients.sort_values(ascending=True)











































def show_barchart():
    # Visualization 2: Bar chart of Coefficients
    coefficients = coefficients.sort_values(ascending=True)
    coefficients = pd.Series(model.coef_, index=X.columns)
    coefficients = coefficients.sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    coefficients.plot(kind='barh')  # Use 'barh' for horizontal bar chart
    plt.title('Coefficients of the Linear Regression Model')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Predictor Variables')
    plt.show()
    return




def show_actual_predicted():
    """
    See the data
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
# def show_MSE():
#     mse = mean_squared_error(Y_test, Y_pred)
#     print("R-squared:", r2)
#     return

# def show_r2():
#     r2 = r2_score(Y_test, Y_pred)
#     print("Mean Squared Error:", mse)
#     return

def show_residuals():
    # RESIDUAL PLOT
    plt.figure(figsize=(10, 6))
    sns.residplot(x=Y_pred, y=Y_test, lowess=True, color="g")
    plt.title('Residual Plot')
    plt.xlabel('Predicted Wins')
    plt.ylabel('Residuals')
    plt.show()
    return


def show_residuals_distribution():
    # distribution plot of residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(Y_test - Y_pred, kde=True)
    plt.title('Distribution of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()
    return

def show_pairplot():
    #Pair PLOT
    sns.pairplot(all_data_1[selected_columns])
    plt.suptitle('Pair Plot of Selected Columns', y=1.02)
    plt.show()
    return

def show_correlation_heatmap():
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