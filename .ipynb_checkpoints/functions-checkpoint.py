import numpy as np

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