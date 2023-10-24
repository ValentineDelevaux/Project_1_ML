import numpy as np
import matplotlib.pyplot as plt
from implementations import *

#CLEANING OF DATA

def standardize_(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    # Vérifiez si std_x est nul (zéro) avant de diviser
    non_zero_std_indices = std_x != 0
    x[:, non_zero_std_indices] = x[:, non_zero_std_indices] / std_x[non_zero_std_indices]
    return x, non_zero_std_indices

def nan_to_median(data):
    #Remplacer les nan par la mediane de chaque colonne
    median = np.nanmedian(data, axis = 0)
    new_data = data
    for col in range(data.shape[1]):
        new_data[:,col] = np.nan_to_num(data[:,col], nan= median[col])
    return new_data

#remplace les nan par la mediane dans les colonnes numeriques
def numeric_nan_to_median(data):
    #Remplacer les nan par la mediane de chaque colonne
    median = np.nanmedian(data, axis = 0)
    new_data = data
    for idx in (separate_column_types(data, max_nb=5)[1]):
        new_data[:,idx] = np.nan_to_num(data[:,idx], nan= median[idx])
    return new_data

#remplace les nan par "Unknown" dans les colonnes categoriques
def categorical_nan_to_unknown(data):
    #Remplacer les nan par la mediane de chaque colonne
    new_data = data
    for idx in (separate_column_types(data, max_nb=5)[0]):
        new_data[:,idx] = np.nan_to_num(data[:,idx], nan= "Unknown")
    return new_data

def remove_zero_std_columns(data):
    std_data, col_to_keep=standardize_(data)
    clean_std_data=std_data[:, col_to_keep]
    return clean_std_data, col_to_keep

def remove_low_variance_features(data, threshold=0.1):
    # Calculate the variance of each feature using NumPy
    feature_variances = np.var(data, axis=1)

    # Identify features with variance above the threshold
    selected_feature_indices = np.where(feature_variances >= threshold)
    selected_features = data[:, selected_feature_indices]

    return selected_features, selected_feature_indices

def remove_high_correlation_feature(data, threshold=0.8):
    correlation_matrix = np.corrcoef(data, rowvar = False)
    plt.figure(figsize=(3, 3))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.show()
    #Remplacer la partie triangulaire basse de la matrice par des 0
    m = np.tril(np.ones(correlation_matrix.shape), k = 0).astype(bool)
    correlation_matrix[m] = 0
    #Creation of a matrix True/False where True == When the correlation is above the treshold
    mask = np.logical_and(np.abs(correlation_matrix) >= threshold, np.abs(correlation_matrix) < 1)
    correlated_features = np.where(mask == True)

    #indices_correlated has the indices of the correlated columns
    col_to_delete = np.unique(correlated_features[0])

    #Removes the correlated columns to x_train_cleaned
    clean_data = np.delete(data, col_to_delete, axis = 1)
    return clean_data, col_to_delete
    

def remove_outlier_rows(data):
    # Calculate the quartiles for each row
    Q1 = np.percentile(data, 5, axis=1)
    Q3 = np.percentile(data, 95, axis=1)
    IQR = Q3 - Q1

    # Define lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter rows based on bounds
    filtered_data = data[(data.min(axis=1) >= lower_bound) | (data.max(axis=1) <= upper_bound), :]
    rows_to_keep = (data.min(axis=1) >= lower_bound) | (data.max(axis=1) <= upper_bound)
    
    return filtered_data, rows_to_keep
'''
def replace_remaining_outlier_values (data, threshold=1.5):
    medians = np.median(data, axis=0)
    for i in range(data.shape[1]):
        col = data[:, i]
        iqr = np.percentile(col, 75) - np.percentile(col, 25)
        lower_bound = np.percentile(col, 25) - threshold * iqr
        upper_bound = np.percentile(col, 75) + threshold * iqr
        outliers = (col < lower_bound) | (col > upper_bound)
        data[outliers, i] = medians[i]
    return data

def replace_remaining_outlier_values(data, threshold=1.5):
    #medians = np.median(data, axis=0)
    data_copy = np.copy(data)
    for i in range(data.shape[1]):
        col = data_copy[:, i]
        median = np.median(col)
        Q1 = np.percentile(col, 5)
        Q3 = np.percentile(col, 95)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        #outliers = (col < lower_bound) | (col > upper_bound)
        #data[outliers, i] = medians[i]
        col[col < lower_bound] = median
        col[col > upper_bound] = median
        
    return data_copy


def rep(data, threshold = 2):
    data_copy = np.copy(data)
    medians = np.median(data_copy, axis = 0)
    q1 = np.percentile(data_copy, 25, axis = 0)
    q3 = np.percentile(data_copy, 75, axis = 0)
    iqr = q3 - q1
    low_thresh = q1 - threshold * iqr
    up_thresh = q3 + threshold * iqr
    for i in range(data_copy.shape[1]):
        data_copy[:, i] = np.where(data_copy[:,i] < low_thresh[i], medians[i], data_copy[:,i])
        data_copy[:, i] = np.where(data_copy[:,i] > up_thresh[i], medians[i], data_copy[:,i])
    return data_copy
'''

def build_model_data(data):
    
    num_samples = data.shape[0]
    #bias_column = np.where(y == -1, 1, 10)
    #data = np.c_[bias_column, data]
    data = np.c_[np.ones(num_samples), data]
    
    return data

def delete_excessive_nan (data, treshold=0.95):
    num_rows = data.shape[0]
    nan_counts = np.sum(np.isnan(data), axis=0)
    keep_mask = nan_counts / num_rows <= threshold
    data_cleaned = data[:, keep_mask]
    deleted_column_indices = np.where(~keep_mask)[0]
    return data_cleaned, deleted_column_indices

def separate_column_types(data, max_nb=5):
    for col_idx in range(data.shape[1]):
        col = data[:, col_idx]
    
        unique_values = np.unique(col)
        num_unique_values = len(unique_values)

        if num_unique_values <= max_nb:
            categorical_columns.append(col_idx)
        else:
            numeric_columns.append(col_idx)

    data_numeric = data[:, numeric_columns]
    data_categorical = data[:, categorical_columns]
    
    return categorical_columns, numeric_columns, data_categorical, data_numerical

#Looking for duplicata
def duplicata (data):
    unq, count = np.unique(data, axis=0, return_counts=True)
    print('Presence of duplicata : ', unq[count].shape != data.shape)
    
#shuffle the rows
def shuffle_rows(xtrain, ytrain):
    shuffler = np.random.permutation(len(xtrain))
    x_train1 = xtrain[shuffler]
    y_train1 = ytrain[shuffler]
    return x_train1, y_train1

#see the distribution of sick and not sick
def test_balancy_dataset (y_train):
    unique_values, counts = np.unique(y_train, return_counts=True)
    freqs = counts/len(y_train)
    # Create bar chart
    labels = unique_values.astype(str)
    std_errors = np.sqrt(freqs*(1-freqs)/len(y_train))
    expected_frequency = 1/len(np.unique(y_train))
    plt.figure(figsize = (2,2))
    plt.bar(labels, freqs, color=['red' if val == -1 else 'blue' for val in unique_values])#, yerr = std_errors*1.96)
    #yerr doesnt work, might have to check why and how to fix it
    plt.axhline(expected_frequency,color='black',linestyle='--')
    plt.xlabel('Values')
    plt.ylabel('Percentage')
    plt.title('Frequency of each value occurrency') 
    plt.show()
    
#pour chaque colonne on enleve les lignes qui sont des outliers
def remove_outliers(xtrain, ytrain, threshold=3):
    
    median = np.median(xtrain, axis=0)
    mad = np.median(np.abs(xtrain - median), axis=0)
    
    # Calculate the modified z-scores
    z_scores = np.abs((xtrain - median) / mad)
    
    # Create a mask for outliers based on the z-scores
    outlier_mask = z_scores > threshold
    
    # Remove rows with outliers in any column
    cleaned_xtrain = xtrain[~np.any(outlier_mask, axis=1)]
    cleaned_ytrain = ytrain[~np.any(outlier_mask, axis=1)]
    
    return cleaned_xtrain, cleaned_ytrain




#SPLITING OF DATA

def split_data(x_train, y_train) :
    #We split x_train in a train part and a validation part
    seed = 12
    ratio = 0.8

    # set seed
    np.random.seed(seed)

    #Number of elements in y
    number = y_train.shape[0]

    index = np.random.permutation(number)

    #Separate into 2 categories at the split
    split = int(np.floor(ratio * len(x_train)))
    train = index[:split]
    test = index[split:]

    #create the new datasets
    x_tr = x_train[train]
    y_tr = y_train[train]
    x_te = x_train[test]
    y_te = y_train[test]
    
    return x_tr, y_tr, x_te, y_te


#Choose best hyperparameters


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_accuracy_ridge(y, x, k_indices, lambda_):
  
    tr_accuracy = []
    te_accuracy = []
    
    for k in range(k_indices.shape[0]):
        
        te_indices = k_indices[k]
        y_te = y[te_indices]
        x_te = x[te_indices]
        tr_indices = (np.delete(k_indices, k, axis=0)).flatten()
        y_tr = y[tr_indices]
        x_tr = x[tr_indices]
        
        

        w, loss=ridge_regression(y_tr, x_tr, lambda_)

        y_te_pred=sigmoid_prediction(x_te, w)
        y_tr_pred=sigmoid_prediction(x_tr, w)
        
        acc_te_k=np.mean(y_te==(y_te_pred == 1))
        acc_tr_k=np.mean(y_tr==(y_tr_pred == 1))
        

        te_accuracy.append(acc_te_k)
        tr_accuracy.append(acc_tr_k)

                                           
    acc_perc_tr = (np.mean(tr_accuracy)*100)
    acc_perc_te = (np.mean(te_accuracy)*100)
   
    return acc_perc_tr, acc_perc_te



def cross_validation_best_lambda_ridge(y, x, k_fold, lambdas):
   
    seed = 12
    k_indices = build_k_indices(y, k_fold, seed)  
    
    tr_accuracy = []
    te_accuracy = []
    
    best_lambda = 1.0
    best_accuracy = 1.0
    best_lamdda_indice = 1.0
    
   
    for i, lambda_ in enumerate(lambdas):
        

        acc_perc_tr = cross_validation_accuracy_ridge(y, x, k_indices, lambda_)[0]
        acc_perc_te = cross_validation_accuracy_ridge(y, x, k_indices, lambda_)[1]
        

        tr_accuracy.append(acc_perc_tr)
        te_accuracy.append(acc_perc_te)
        

        if(acc_perc_te >= max(te_accuracy) and len(te_accuracy) != 0):
            best_lambda_indice = i
            

    best_lambda = lambdas[best_lambda_indice]
    best_accuracy = te_accuracy[best_lambda_indice]


    cross_validation_visualization_lambda_ridge(lambdas, accuracy_tr, accuracy_te)
    

    print(f"The best lambda is {best_lambda} corresponding to {best_accuracy}% of accuracy")

    return best_lambda, best_accuracy


def cross_validation_visualization_lambda_ridge(lambdas, tr_accuracy, te_accuracy):

    plt.semilogx(lambdas, tr_accuracy, marker=".", color='b', label='train accuracy')
    plt.semilogx(lambdas, te_accuracy, marker=".", color='r', label='test accuracy')
    plt.xlabel("gamma")
    plt.ylabel("accuracy")
    plt.title("cross validation best gamma")
    plt.legend()
    #plt.savefig("cross_validation_best_lambda")


    
# Prediction

def sigmoid_prediction (x, w, threshold=0.5):
    #y_pred_proba has the proba of each y to be 1 or -1
    y_pred_proba = sigmoid (np.dot(x, w))
    #If proba > 0.5, y becomes 1, else -1
    y_pred = (y_pred_proba >= threshold).astype(int)
    y_pred[y_pred == 0] = -1
    return y_pred