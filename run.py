#File for the final run, that gives the best prediction possible with our algorithm

#Importations
from implementations import *
from helpers import *
import matplotlib.pyplot as plt
import numpy as np
from functions import *

#Loading of data
x_train, x_test, y_train, train_ids, test_ids = load_csv_data('dataset_to_release/', sub_sample=False)

#Cleaning of data
x_train1, y_train1, x_test1 = clean_data (x_train, y_train, x_test)

#Expansion of data
categorical_columns, numeric_columns, _, _ = separate_column_types(x_train1, max_nb=10)

x_train1=build_numerical_cos(x_train1, categorical_columns, numeric_columns)
x_test1=build_numerical_cos(x_test1, categorical_columns, numeric_columns)

#Spliting of data
x_tr, y_tr, x_te, y_te = split_data(x_train1, y_train1)

#Creation of balanced subsets
sub_y_trains, sub_x_trains = creation_subsets(y_tr, x_tr)

#Initial w with least squares
weights = []
for i in range(len(sub_x_trains)):
    w,_ = least_squares(sub_y_trains[i], sub_x_trains[i])
    weights.append(w)
    
w_initial = np.mean(weights, axis = 0)

#Gradient descent with regularized logistic regression and the best hyperparameters found in tests
max_iters = 15
lambda_ = 0.13333333333333333
gamma = 0.022222222222222223

weights_sub = []
for a in range(len(sub_x_trains)):
    w = w_initial
    w, _ = reg_logistic_regression(sub_y_trains[a], sub_x_trains[a], lambda_, initial_w, max_iters, gamma)
    weights_sub.append(w)

w_final = np.mean(weights_sub, axis = 0)

#Best threshold calculation
best_threshold = compute_threshold(w_final, x_te, y_te)

#Prediction generation
y_pred = sigmoid_prediction_(x_test1, w_final, threshold = best_threshold)

#Submission file generation
name = 'submission'
create_csv_submission(test_ids, y_pred, name)