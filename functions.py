import numpy as np
import matplotlib.pyplot as plt
from implementations import *

#----------------------------------------------------------------------------------

#CLEANING OF DATA

def duplicata (data):   
    """ 
    Prints an error message if the data contains duplicate rows.
    ------------

    Arguments:
    data : array, shape=(N,D) 
        
    ------------
    Return :
    no return
    
    """        
    unq, count = np.unique(data, axis=0, return_counts=True)   
    if ((unq[count].shape != data.shape)==True):        
        print("The data contain duplicate rows")
    

    

def shuffle_rows(xtrain, ytrain):
    """ 
    Shuffles the rows of xtrain and ytrain at the same time to keep them aligned.
    ------------

    Arguments:
    x_train : array, shape=(N,D) Array to be shuffled
    y_train : array, shape=(N,)  Array to be shuffled
    
    ------------
    Return: 
    x_train1 : array, shape=(N,D) Array with shuffled rows
    y_train1 : array, shape=(N,) Array with shuffled rows
    
    """
    shuffler = np.random.RandomState(seed=42).permutation(len(xtrain))
    x_train1 = xtrain[shuffler]
    y_train1 = ytrain[shuffler]
    return x_train1, y_train1


def test_balancy_dataset (y_train):   
    """ 
    Tests the balancy of the dataset between 1 and -1.
    ------------

    Arguments:
    
    y_train : array, shape=(N,)  Array containing -1 and 1 corresponding to the medical diagnostic of each individual of the dataset
        
    ------------
    Return: 
    no return (prints a bar chart showing frequency of 1 and -1 in y_train)
    
    """
    unique_values, counts = np.unique(y_train, return_counts=True)
    freqs = counts/len(y_train)
    # Create bar chart
    #labels = unique_values.astype(str)
    #std_errors = np.sqrt(freqs*(1-freqs)/len(y_train))
    #expected_frequency = 1/len(np.unique(y_train))
    #plt.figure(figsize = (2,2))
    #plt.bar(labels, freqs, color=['red' if val == -1 else 'blue' for val in unique_values])
    #plt.axhline(expected_frequency,color='black',linestyle='--')
    #plt.xlabel('Values')
    #plt.ylabel('Percentage')
    #plt.title('Frequency of each value occurrency') 
    #plt.show()
    
def nan_to_median(data):  
    """ 
    Replaces all Nan by the median of the corresponding feature
    ------------

    Arguments:
    data : array, shape=(N,D) 
        
    ------------
    Return: 
    new_data : array, shape=(N,D) Array with all the Nan replaced by the median
    
    """
    #Remplacer les nan par la mediane de chaque colonne
    median = np.nanmedian(data, axis = 0)
    new_data = data
    for col in range(data.shape[1]):
        new_data[:,col] = np.nan_to_num(data[:,col], nan= median[col])
    return new_data

    
def standardize_(x):   
    """ 
    Standardizes the columns of x.
    ------------

    Arguments:
    x : array, shape=(N,D) Array to be standardized
        
    ------------
    Return: 
    x : array, shape=(N,D) Array with all the standardized columns
    non_zero_indices : list with indices of columns with standard deviation !=0
    
    """
    
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    # checks if std_x null before division
    non_zero_std_indices = std_x != 0
    x[:, non_zero_std_indices] = x[:, non_zero_std_indices] / std_x[non_zero_std_indices]
    return x, non_zero_std_indices


def remove_zero_std_columns(data):    
    """ 
    Standardizes a matrix and removes the columns with 0 standard deviation.
    ------------

    Arguments:
    data : array, shape=(N,D) Array to be standardized and modified
        
    ------------
    Return: 
    clean_std_data : array, shape=(N,E) Array with all the standardized columns, and without columns with 0 std
    col_to_keep : list, indices of columns that were kept in x
    
    """
    #uses the function strandardize_ to standardize the matrix and collect indices of features with  
    #zero std
    std_data, col_to_keep=standardize_(data)
    #removes the features corresponding to those indices
    clean_std_data=std_data[:, col_to_keep]
    #returns the indices of the columns that were kept to tranform x_test in the same way
    return clean_std_data, col_to_keep


def remove_high_correlation_feature(data, threshold=0.8):   
    """ 
    Removes feature that are very correlated in data by computing a correlation matrix.
    ------------

    Arguments:
    data : array, shape=(N,D) Array that needs to be cleaned
    treshold : int, criteria used to determine whether the correlation is too strong or not
        
    ------------
    Return: 
    clean_data : array, shape=(N,E) data array without the columns that had too strong correlation
    col_to_delete : list, indices of columns that were deleted in x
    
    """
    correlation_matrix = np.corrcoef(data, rowvar = False)
    correlation_matrix = np.round(correlation_matrix, 5)
    #plt.figure(figsize=(3, 3))
    #plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    #plt.colorbar()
    #plt.show()
    
    #we transform the correlation matrix into a triangular matrix to avoid removing too much features
    m = np.tril(np.ones(correlation_matrix.shape), k = 0).astype(bool)
    correlation_matrix[m] = 0
    #creates a True/False matrix: True means that the correlation is above the threshold
    mask = np.logical_and(np.abs(correlation_matrix) >= threshold, np.abs(correlation_matrix) < 1)
    correlated_features = np.where(mask == True)
    
    #selects all the indices of the features that need to be deleted
    col_to_delete = np.unique(correlated_features[0])

    #removes the correlated columns in data
    clean_data = np.delete(data, col_to_delete, axis = 1)
    
    return clean_data, col_to_delete


def build_model_data(data):   
    """ 
    Adds a column of ones to the x array for the bias term.
    ------------

    Arguments:
    data : array, shape=(N,E) 

    ------------
    Return: 
    data : array, shape=(N,E+1) data array without the columns of ones added
    
    """    
    num_samples = data.shape[0]
    data = np.c_[np.ones(num_samples), data]
    
    return data

def clean_data (x_train, y_train, x_test):
    
    """Cleans the data for model training and testing (see comments below for more details).

    Args:
        x_train: shape=(N,D) The data for training
        y_train: shape=(N,)  The results for training
        x_test: shape=(M,D)  The data for prediction

    Returns:
        The cleaned arrays x_train_clean, y_train_clean and x_test_clean.
    """
    
    # checks if x_train contains duplicate rows
    duplicata(x_train) 
    
    # shuffles x_train and y_train at the same time to avoid mixing the rows
    x_train_clean, y_train_clean = shuffle_rows(x_train, y_train)
    
    # deletes 9 first columns of x_train and x_test because they contain unuseful informations
    x_train_clean= x_train_clean[:,9:]
    x_test_clean = x_test[:,9:]
    
    # replaces all Nan by the mean of the feature in x_train and x_test 
    x_train_clean=nan_to_median(x_train_clean)
    x_test_clean=nan_to_median(x_test_clean)
    
    # standardizes the columns of x_test
    x_test_clean, _ = standardize_(x_test_clean)
    
    # standardizes the columns of x_train and deletes the columns that have 0 standard deviation
    x_train_clean, col_to_keep=remove_zero_std_columns(x_train_clean)
    # deletes the corresponding features in x_test
    x_test_clean= x_test_clean[:, col_to_keep]
    
    print(len(x_train_clean[1]))
    # deletes features with high correlation in x_train and removes the corresponding features in x_test
    x_train_clean, col_to_delete=remove_high_correlation_feature(x_train_clean)
    x_test_clean= np.delete(x_test_clean, col_to_delete, axis = 1)
    print(len(x_train_clean[1]))
    # adds a column of one to the features for the bias term
    x_train_clean = build_model_data(x_train_clean)
    x_test_clean = build_model_data(x_test_clean)
    
    print('Shape of x_train after cleaning: ', x_train_clean.shape)
    print('Shape of y_train after cleaning: ', y_train_clean.shape)
    print('Shape of x_test after cleaning: ', x_test_clean.shape)
    
    return x_train_clean, y_train_clean, x_test_clean



#----------------------------------------------------------------------------------

#EXPAND FEATURES


def separate_column_types(data, max_nb=5):
    """ 
    Defines which columns of the data matrix correspond to numerical and categorical features.
    ------------

    Arguments:
    data : shape=(N,D) 
    max_nb : int, minimum number of different values in the column required to be considered a numerical feature
    

    ------------
    Return: 
    categorical_columns: list, indices corresponding to features considered categorical
    numeric_columns: list, indices corresponding to features considered numeric
    data_categorical: array, only the categorical features
    data_numeric: array, only the numerical features
    
    """  
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




def build_numerical_cos(x, categorical_columns, numeric_columns):
    """
    For each feature considered as numerical, add a new corresponding column filled with the cosinus of these values
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
    new_num_features = num_features + len(numeric_columns) 
    new_x = np.copy(x)
        
    for idx in range(num_features):
        if idx in numeric_columns:
            new_x = np.concatenate((new_x, np.cos(new_x[:, idx]).reshape(-1,1)), axis = 1)
           
    return new_x


#--------------------------------------------------------------------------------------

#SPLIT DATA

def split_data(x_train, y_train) :
    """
    Split the dataset based on a split ratio. If ratio is 0.8 you will have 80% of your data set dedicated to training
    and the rest dedicated to testing.
    ------------
    
    Arguments :
    x : numpy array of shape (N, D), N is the number of samples, D the number of features.
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
    
    #mixing samples of 1 and -1 and shuffling the rows to obtain balanced datasets
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

#PREDICTION

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
    #elements in y_pred_proba correspond to the probability of each sample to be 1 or -1
    y_pred_proba = sigmoid (np.dot(x, w))
    #If proba > threshold, y becomes 1, else -1
    y_pred = (y_pred_proba >= threshold).astype(int)
    y_pred[y_pred == 0] = -1
    return y_pred


#----------------------------------------------------------------------------------

#TESTING CALCULATIONS

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
        raise ValueError("Both entries have to be of the same size.")

    # Counts the number of correct predictions
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)

    # Computes the accuracy percentage
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

def compute_threshold( w, x, y):
    f1_score = []
    ts = []
    thresholds = np.linspace(0, 1, 100)

    for t in thresholds:
        y_pred = sigmoid_prediction_(x, w, t)
        f1_score.append( compute_f1(y, y_pred))
        ts.append(t)
        
    best_index = f1_score.index(max(f1_score))
    print('Threshold : ', ts[best_index])
    threshold = ts[best_index]
    
    return threshold
    

def test(w, x_te, y_te, losses = None, f1_scores = None, accuracies = None, gammas = None, lambdas = None):
    """
    Finds the best threshold that optimizes the f1 score and display the accuracy and the F1 score of the tested arguments
    ------------
    
    Arguments : 
    w : array, weights, shape (D, )
    x_te : array, tested dataset, shape = (N, D)
    y_te : array, tested dataset, shape = (N, ) 
    
    ------------
    Return :
    None
    """
    
    f1_score = []
    ts = []
    thresholds = np.linspace(0, 1, 100)

    for t in thresholds:
        y_pred = sigmoid_prediction_(x_te, w, t)
        f1_score.append( compute_f1(y_te, y_pred))
        ts.append(t)
        
    best_index = f1_score.index(max(f1_score))
    print('Threshold : ', ts[best_index])
    threshold = ts[best_index]
    
    y_pred = sigmoid_prediction_ (x_te, w, threshold=threshold) 
    print('Accuracy : ', accuracy(y_te, y_pred))
    print('F1 score : ', compute_f1(y_te, y_pred))
    """
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
            
    """
    