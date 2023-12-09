import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# import hoopin
# Assuming you have already loaded your data and defined the variables as mentioned in your previous code


all_data_1 =  pd.read_csv("/datasets/basketball.csv")
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



coefficients = pd.Series(model.coef_, index=X.columns)
sorted_coefficients = coefficients.abs().sort_values(ascending=False)
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
    



def show_heatmap():
    plt.figure(figsize=(10, 8))
    sns.heatmap(all_data_1[selected_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Selected Columns')
    plt.show()

    """Applies _fit_tranform to the data, x, y, and returns the RF-PHATE embedding

    x : {array-like, sparse matrix} of shape (n_samples, n_features)
        The training input samples. Internally, its dtype will be converted to dtype=np.float32.
        If a sparse matrix is provided, it will be converted into a sparse csc_matrix.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The target values (class labels in classification, real numbers in regression).
        
    x_test : {array-like, sparse matrix} of shape (n__test_samples, n_features)
        An optional test set. The training set buildes the RF-PHATE model, but the 
        embedding can be extended to this test set.


    Returns
    -------
    array-like (n_features, n_components)
        A lower-dimensional representation of the data following the RF-PHATE algorithm
    """
            
    return 