import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd




def show_heatmap(all_data_1=[1,2,3],selected_columns=['3P%','2P%', 'AST','TRB','STL',"TS%", 'W', 'L']):
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