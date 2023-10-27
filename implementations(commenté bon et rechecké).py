import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------
#MANDATORY FUNCTIONS

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    The Gradient Descent (GD) algorithm.
    
    ------------
    Arguments:
    y : array, shape = (N, )
    tx : array, shape = (N, D)
    initial_w : shape = (D, ). The initial guess (or the initialization) for the model parameters
    max_iters : a scalar denoting the total number of iterations of GD
    gamma : a scalar denoting the stepsize
    
    ------------
    Return :
    loss : scalar, loss value for the last iteration of GD
    w : array, shape = (N, ). The vector of model parameters after the last iteration of GD
    """

    w = initial_w
    for n_iter in range(max_iters):
        loss =  compute_loss(y, tx, w)
        gradient = compute_gradient(y, tx, w)
        w = w - np.dot(gamma, gradient)

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, batch_size, max_iters, gamma):
    """
    Stochastic Gradient Descent algorithm (SGD).
    ------------

    Arguments:
    y : shape = (N, )
    tx : shape = (N, D)
    initial_w : shape=(2, ). The initial guess (or the initialization) for the model parameters
    batch_size : a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
    max_iters : a scalar denoting the total number of iterations of SGD
    gamma : a scalar denoting the stepsize
    
    ------------
    Return:
    loss : scalar, loss value for the last iteration of SGD
    w : array, shape = (N, ). The vector of model parameters after the last iteration of SGD
    """
    w = initial_w
            
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
 
            w = w - np.multiply(gamma, compute_stoch_gradient(minibatch_y, minibatch_tx, w))
            loss = compute_loss(minibatch_y, minibatch_tx, w)
        
    return w, loss


def least_squares(y, tx):
    """
    Calculate the least squares solution
    ------------

    Arguments:
    y : array, shape = (N,), N is the number of samples.
    tx : array, shape = (N, D), D is the number of features.

    ------------
    Return:
    w : optimal weights, numpy array of shape(D,), D is the number of features.
    loss : scalar, loss value computed with MSE
    """
    
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    
    w = np.linalg.solve(a,b)
    
    loss = compute_loss(y, tx, w)
    
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Implement ridge regression algorithm
    ------------
    
    Arguments:
    y : array, shape = (N, ), N is the number of samples.
    tx : array, shape = (N, D), D is the number of features.
    lambda_ : scalar.
    
    ------------
    Return:
    w : array, shape = (D, ), optimal weights
    loss : scalar, loss value computed with MSE
    """
    lambda_prime = 2 * tx.shape[0] * lambda_
    
    w = np.linalg.solve(tx.T.dot(tx) + lambda_prime * np.identity(tx.shape[1]), tx.T.dot(y)  )
    loss = compute_loss(y, tx, w)

    return w, loss


def logistic_regression(y, tx, w, gamma):
    """
    Compute one step of gradient descent using logistic regression
    ------------

    Arguments :
    y : array, shape = (N, )
    tx : shape = (N, D)
    w :  shape=(D, )
    gamma: float
    
    ------------
    Return :
    loss: scalar, loss value computed with logistic loss
    w : array, shape = (D, ), final w after the logistic regression
    """

    loss = calculate_logistic_loss(y, tx, w)
    gradient = calculate_logistic_gradient(y, tx, w)
    w = w - gamma * gradient
    
    return w, loss


def reg_logistic_regression(y, tx, w, gamma, lambda_):
    """
    Compute one step of gradient descent, using the penalized logistic regression.

    ------------
    Arguments :
    y : shape = (N, )
    tx : shape = (N, D)
    w : shape = (D, )
    gamma : scalar
    lambda_ : scalar

    ------------
    Return :
    w : shape = (D, )
    loss : scalar number
    """

    loss, gradient = compute_reg_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * gradient

    return w, loss


#------------------------------------------------------------------------------------------
#OTHER USEFUL FUNCTIONS

def compute_loss(y, tx, w):
    """
    Calculate the loss using Mean Square Error
    ------------
    
    Arguments :
    y : array, shape = (N, )
    tx : array, shape = (N,D)
    w : array, shape = (D,). The vector of model parameters.

    ------------
    Return:
    loss : value of the loss (a scalar), corresponding to the input parameter w.
    """
    
    loss = (1/ 2) * np.mean((y-tx.dot(w))**2)
   
    return loss


def compute_loss_pred(y, y_pred):
    """
    Calculate the loss between the predicted value and the true value
    ------------
    
    Arguments :
    y : array, shape = (N, )
    y_pred : array, shape = (N,), predicted values from a model
    S
    ------------
    Return :
    loss : value of the loss (a scalar)
    """
    
    loss = (1/ 2) * np.mean((y-y_pred)**2)
   
    return loss


def compute_gradient(y, tx, w):
    """
    Computes the gradient at w
    ------------

    Arguments :
    y : array, shape = (N, )
    tx : array, shape = (N, D)
    w : array, shape = (N, ). The vector of model parameters.

    ------------
    Return :
    gradient : an array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    
    error = y - tx.dot(w)
    gradient = (-1/y.shape[0]) * tx.T.dot(error)
    
    return gradient    





## MEAN SQUARED ERROR STOCHASTIC GRADIENT

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    ------------
    
    Arguments :
    y :array, shape = (N, )
    tx : array, shape = (N, D)
    batch_size : scalar, the size of each batch
    num_batches : int, number of batches
    shuffle : bool, shuffle the dataset if True
    
    ------------
    Return : 
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def compute_stoch_gradient(y, tx, w):
    """
    Compute a stochastic gradient
    ------------
    
    Arguments:
    y : shape = (N, )
    tx : shape = (N, D)
    w : shape = (D, ). The vector of model parameters.

    ------------
    Return:
    stoch_gradient : array, shape = (D, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    
    error = y - tx.dot(w)
    stoch_gradient = (-1 / y.shape[0]) * tx.T.dot(error)
    
    return stoch_gradient


## LEAST SQUARES

def least_squares(y, tx):
    """
    Calculate the least squares solution
    ------------

    Arguments:
    y : array, shape = (N,), N is the number of samples.
    tx : array, shape = (N, D), D is the number of features.

    ------------
    Return:
    w : optimal weights, numpy array of shape(D,), D is the number of features.
    loss : scalar, loss value computed with MSE
    """
    
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    
    w = np.linalg.solve(a,b)
    
    loss = compute_loss(y, tx, w)
    
    return w, loss


## LOGISTIC REGRESSION

def sigmoid(t):
    """
    Apply sigmoid function on t.
    ------------
    
    Arguments :
    t : scalar or numpy array
    
    ------------
    Return :
    scalar or numpy array
    """
    
    return 1/ (1 + np.exp(-t))

def calculate_logistic_loss(y, tx, w):
    """
    Compute the cost by negative log likelihood
    ------------
    
    Arguments :
    y : array, shape = (N, )
    tx : array, shape = (N, D)
    w :  array, shape = (D, )
    
    ------------
    Return :
    loss : scalar, non-negative loss value
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    y_predicted = sigmoid(tx.dot(w))
        
    loss = np.linalg.norm((-1/(tx.shape[0]))*(y.T.dot(np.log(y_predicted))+(1-y).T.dot(np.log(1-y_predicted))))

    return loss

def calculate_logistic_gradient(y, tx, w):
    """
    Compute the gradient of loss.
    ------------
    
    Arguments :
    y :  array, shape = (N, )
    tx : array, shape = (N, D)
    w :  array, shape = (D, )
    
    ------------
    Returns :
    grad : array, shape (D, 1), logistic gradient
    """
    
    y_predicted = sigmoid(tx.dot(w))
    grad = (1 / (tx.shape[0])) * tx.T.dot(y_predicted - y)
    
    return grad
    

## REGULARIZED LOGISTIC REGRESSION

def compute_reg_logistic_regression(y, tx, w, lambda_):
    """
    Return the loss and gradient.
    ------------
    
    Arguments :
    y :  shape = (N, )
    tx : shape = (N, D)
    w :  shape = (D, )
    lambda_ : scalar
    
    ------------
    Return :
    loss : scalar, loss value computed with MSE
    gradient : array, shape = (D, 1)
    """
    
    loss = compute_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w) )
    gradient = compute_gradient(y, tx, w) +  2 * lambda_ * w 
    
    return loss, gradient

    
