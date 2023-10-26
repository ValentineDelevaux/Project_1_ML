import numpy as np
import matplotlib.pyplot as plt
from implementations import *

#----------------------------------------------------------------------------------

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
    numeric_columns=[]
    categorical_columns=[]
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
    
    return categorical_columns, numeric_columns, data_categorical, data_numeric

#Looking for duplicata
def duplicata (data):
    unq, count = np.unique(data, axis=0, return_counts=True)
    print('Presence of duplicata : ', unq[count].shape != data.shape)
    

def shuffle_rows(xtrain, ytrain):
    """
    Shuffle the rows of the dataset
    
    Arguments : 
    xtrain: array, shape = (N, D)
    ytrain: array, shape = (N, )
    
    ------------
    Return:
    x_train1 : array, shuffled xtrain, shape = (N, D) 
    y_train1 : array, shuffled ytrain, shape = (N, )
    """
    
    shuffler = np.random.permutation(len(xtrain))
    x_train1 = xtrain[shuffler]
    y_train1 = ytrain[shuffler]
    
    return x_train1, y_train1

def test_balancy_dataset (y_train):
    """
    Display a plot of the distribution of sick and non-sick individuals among the dataset
    ------------
    
    Arguments : 
    y_train : array, shape = (N, )
    
    ------------
    Return : 
    """
    unique_values, counts = np.unique(y_train, return_counts=True)
    freqs = counts/len(y_train)
    
    # Create bar chart
    labels = unique_values.astype(str)
    expected_frequency = 1/len(np.unique(y_train))
    
    plt.figure(figsize = (2,2))
    plt.bar(labels, freqs, color=['red' if val == -1 else 'blue' for val in unique_values])    
    plt.axhline(expected_frequency,color='black',linestyle='--')
    plt.xlabel('Values')
    plt.ylabel('Percentage')
    plt.title('Frequency of each value occurrency') 
    plt.show()
    
#--------------------------------------------------------------------------------------
#SPLITING OF DATA

def split_data(x_train, y_train) :
    """
    Split the dataset based on a split ratio. If ratio is 0.8 you will have 80% of your data set dedicated to training
    and the rest dedicated to testing.
    ------------
    
    Arguments :
    x : numpy array of shape (N,), N is the number of samples.
    y : numpy array of shape (N,).
    
    ------------
    Returns :
    x_tr : numpy array containing the train data.
    x_te : numpy array containing the test data.
    y_tr : numpy array containing the train labels.
    y_te : numpy array containing the test labels.
    """
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


def creation_subsets(y, x):
    '''
    Split a dataset in multiple balanced subsets
    ------------
    
    Arguments : 
    y : array, shape = (N, )
    x : array, shape = (N, D)
    
    ------------    
    Return:
    balanced_x_train : list of balanced subsets created from x 
    balanced_y_train : list of balanced subsets created from y
    
    Both lists have the same length and correspond to each other
    '''
    #number of sick people in y_train
    nb_ones = np.count_nonzero(y == 1)  
    
    #number of subset to be created
    nb_subsets = np.round( x.shape[0] / nb_ones, 0)

    indices_ones = np.argwhere(y == 1)
    
    #train dataset of the sick people
    x_train_ones = x[indices_ones]
    x_train_ones_ = np.squeeze(x_train_ones, axis=1)
    y_train_ones = y[indices_ones]
    y_train_ones_ = np.squeeze(y_train_ones, axis = 1)

    indices_minus_ones = np.argwhere(y == -1)
    
    #train dataset of the non-sick people
    x_train_minus_ones = x[indices_minus_ones]
    x_train_minus_ones_ = np.squeeze(x_train_minus_ones, axis=1)
    y_train_minus_ones = y[indices_minus_ones]
    
    #splitting of the dataset in subsets
    split_x_train_minus_ones = np.array_split(x_train_minus_ones, nb_subsets)
    split_y_train_minus_ones = np.array_split(y_train_minus_ones, nb_subsets)

    balanced_x_train = [] 
    balanced_y_train = []

    for i in range(len(split_x_train_minus_ones)):
        together_x = np.concatenate((split_x_train_minus_ones[i], x_train_ones), axis=0)
        together_x = np.squeeze(together_x, axis=1)
    
        together_y = np.concatenate((split_y_train_minus_ones[i], y_train_ones), axis = 0)
        together_y = np.squeeze(together_y, axis=1)
        together_x, together_y = shuffle_rows (together_x, together_y)
    
        balanced_x_train.append(together_x) 
        balanced_y_train.append(together_y)
    
    return balanced_y_train, balanced_x_train 

#----------------------------------------------------------------------------------
#Creation new columns

def build_numerical_cos(x, categorical_columns, numerical_columns):
    """
    For each column with numerical value, add a new corresponding column filled with the cosinus of these values
    ------------
    
    Arguments : 
    x : array, shape = (N, D) (N = number of sample, D = number of feature)
    categorical_columns : list, index of the columns containing categorical values
    numerical_columns : list, index of the columns containing numerical values
    
    ------------
    Return : 
    new_x : new training dataset with the additional columns, shape = (N, D+number of numerical columns)
    """

    num_samples, num_features = x.shape
    new_num_features = num_features + len(numerical_columns) 
    new_x = np.copy(x)
        
    for idx in range(num_features):
        if idx in numerical_columns:
            new_x = np.concatenate((new_x, np.cos(new_x[:, idx]).reshape(-1,1)), axis = 1)
           
    return new_x

#----------------------------------------------------------------------------------
#Validation

def build_k_indices(y, k_fold, seed):
    """
    Build k indices for k-fold.
    ------------
    
    Arguments :
    y: shape = (N,)
    k_fold: K in K-fold, i.e. the fold num
    seed: the random seed
    
    ------------
    Returns:
    A 2D array of shape = (k_fold, N/k_fold) that indicates the data indices for each fold

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

#-------------------------------------------------------------------------------------------    
#Prediction

def sigmoid_prediction_ (x, w, threshold=0.5):
    """
    Prediction of the model with the sigmoid function after its training 
    ------------
    
    Arguments : 
    x : array, test dataset, shape = (N, D)
    w : array, weights, shape = (D, )
    threshold : scalar, between 0 and 1, sensitivity of the system to assess who is sick and who isn't
    
    ------------
    Return : 
    y_pred : predicted values computed with the sigmoid function 
    """
    #y_pred_proba has the proba of each y to be 1 or -1
    y_pred_proba = sigmoid (np.dot(x, w))
    
    #If proba > 0.5, y becomes 1, else -1
    y_pred = (y_pred_proba >= threshold).astype(int)
    y_pred[y_pred == 0] = -1
    
    return y_pred

#-------------------------------------------------------------------------------------------
#Test calculations

def accuracy(y_true, y_pred):
    """
    Compute the accuracy of a classification model
    ------------
    
    Arguments :
    y_true : array, true values from the dataset, shape = (N, ) 
    y_pred : array, values predicted by the model, shape = (N, )
    
    ------------
    Returns:
    accuracy : percentage, accuracy of the classification model 
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Les entrées doivent avoir la même longueur.")

    #count of the correctly predicted values
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)

    #computation of the accuracy as a percentage 
    accuracy = (correct_predictions / len(y_true)) * 100.0

    return accuracy

def compute_f1(y_true, y_pred):
    """
    Compute the F1 score of the model
    ------------
    
    Arguments : 
    y_true : array, true values from the dataset, shape = (N, ) 
    y_pred : array, values predicted by the model, shape = (N, )
    
    ------------
    Return : 
    f1 : scalar between 0 and 1, F1 score of the model
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == -1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == -1))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP+ FN) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def test(w, x_te, y_te):
    """
    Display the accuracy and the F1 score of the tested arguments
    ------------
    
    Arguments : 
    w : array, weights, shape (D, )
    x_te : array, tested dataset, shape = (N, D)
    y_te : array, tested dataset, shape = (N, ) 
    
    ------------
    Return :
    """
    y_pred = sigmoid_prediction_ (x_te, w, threshold=threshold) 
    print('Accuracy : ', accuracy(y_te, y_pred))
    print('F1 score : ', compute_f1(y_te, y_pred))
