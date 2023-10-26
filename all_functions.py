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


def remove_numerical_outliers(x_train, y_train, columns_to_check, z_threshold=3):
    # Copy the input array to avoid modifying the original array
    modified_x_train = x_train.copy()

    modified_y_train = y_train.copy()

    outlier_rows = []
    
    # Iterate through the specified columns
    for col in columns_to_check:
        column_data = modified_x_train[:, col]

        # Calculate the z-scores for the column
        z_scores = np.abs((column_data - np.mean(column_data)) / np.std(column_data))

        # Find rows with z-scores greater than the threshold
        outlier_rows_col = np.where(z_scores > z_threshold)[0]
        outlier_rows.extend(outlier_rows_col)

    # Remove duplicate row indices and sort them
    outlier_rows = list(set(outlier_rows))
    outlier_rows.sort()

    # Remove the outlier rows from the modified array and vector
    modified_x_train = np.delete(modified_x_train, outlier_rows, axis=0)

    modified_y_train = np.delete(modified_y_train, outlier_rows, axis=0)

    # Return the modified array and vector
    return modified_x_train, modified_y_train



#--------------------------------------------------------------------------------------

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


def creation_subsets(y, x):
    '''
    Return 2 lists (y and x) of subsets of the same size
    '''
    nb_ones = np.count_nonzero(y == 1)  #28 975 sick people in y_train
    nb_subsets = np.round( x.shape[0] / nb_ones, 0) #11 groups

    indices_ones = np.argwhere(y == 1)
    x_train_ones = x[indices_ones] #x_train des gens malades
    x_train_ones_ = np.squeeze(x_train_ones, axis=1) #Enleve une dimension en trop
    y_train_ones = y[indices_ones]
    y_train_ones_ = np.squeeze(y_train_ones, axis = 1)

    indices_minus_ones = np.argwhere(y == -1)
    x_train_minus_ones = x[indices_minus_ones] #x_train des gens non malades
    x_train_minus_ones_ = np.squeeze(x_train_minus_ones, axis=1) #Enleve une dimension en trop
    y_train_minus_ones = y[indices_minus_ones]

    split_x_train_minus_ones = np.array_split(x_train_minus_ones, nb_subsets)  # 11 groups of 27 197 or 27 196 people
    split_y_train_minus_ones = np.array_split(y_train_minus_ones, nb_subsets)

    balanced_x_train = [] #List of 11 x_train
    balanced_y_train = [] #List of 11 y_train

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

#Validation

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


#-------------------------------------------------------------------------------------------    

# Prediction

def sigmoid_prediction (y, x, w, threshold=0.5):
    #y_pred_proba has the proba of each y to be 1 or -1
    y_pred_proba = sigmoid (np.dot(x, w))
    #If proba > 0.5, y becomes 1, else -1
    y_pred = (y_pred_proba >= threshold).astype(int)
    y_pred[y_pred == 0] = -1
    loss = compute_loss_pred(y, y_pred_proba)
    return y_pred, loss

def sigmoid_prediction_ (x, w, threshold=0.5):
    #y_pred_proba has the proba of each y to be 1 or -1
    y_pred_proba = sigmoid (np.dot(x, w))
    #If proba > 0.5, y becomes 1, else -1
    y_pred = (y_pred_proba >= threshold).astype(int)
    y_pred[y_pred == 0] = -1
    return y_pred

def accuracy(y_true, y_pred):
    """
    Calcule l'exactitude (accuracy) d'un modèle de classification.

    Args:
    - y_true : Liste ou tableau NumPy des vraies étiquettes (true labels).
    - y_pred : Liste ou tableau NumPy des étiquettes prédites (predicted labels).

    Returns:
    - La valeur de l'exactitude en pourcentage (de 0 à 100).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Les entrées doivent avoir la même longueur.")

    # Comptez le nombre de prédictions correctes
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)

    # Calculez l'exactitude en pourcentage
    accuracy = (correct_predictions / len(y_true)) * 100.0

    return accuracy

def compute_f1(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == -1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == -1))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP+ FN) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def test(w, x_te, y_te, losses = None, f1_scores = None, accuracies = None, gammas = None, lambdas = None, threshold = 0.5):
    y_pred = sigmoid_prediction_ (x_te, w, threshold=threshold) 
    print('Accuracy : ', accuracy(y_te, y_pred))
    print('F1 score : ', compute_f1(y_te, y_pred))
    
    if losses is not None:
        if gammas is not None:
            plt.figure(figsize = (2,2))
            plt.xlabel('gammas')
            plt.ylabel('losses')
            plt.plot(gammas, losses)
            plt.show()
        if lambdas is not None:
            plt.figure(figsize = (2,2))
            plt.xlabel('lambdas')
            plt.ylabel('losses')
            plt.plot(lambdas, losses)
            plt.show()
            
    if f1_scores is not None:
        if gammas is not None:
            plt.figure(figsize = (2,2))
            plt.xlabel('gammas')
            plt.ylabel('F1')
            plt.plot(gammas, f1_scores)
            plt.show()
        if lambdas is not None:
            plt.figure(figsize = (2,2))
            plt.xlabel('lambdas')
            plt.ylabel('F1')
            plt.plot(lambdas, f1_scores)
            plt.show()
            
    if accuracies is not None:
        if gammas is not None:
            plt.figure(figsize = (2,2))
            plt.xlabel('gammas')
            plt.ylabel('Accuracy')
            plt.plot(gammas, accuracies)
            plt.show()
        if lambdas is not None:
            plt.figure(figsize = (2,2))
            plt.xlabel('lambdas')
            plt.ylabel('Accuracy')
            plt.plot(lambdas, accuracies)
            plt.show()            
    