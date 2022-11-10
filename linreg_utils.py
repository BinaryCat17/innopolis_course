import numpy as np

def predict(X, W):
    return X * W.T

def least_squares(X, Y):
    return (np.linalg.inv((X.T * X)) * X.T * Y).T

def MSE(Y, Y_true):
    return np.mean(np.power(Y - Y_true, 2))

def RMSE(Y, Y_true):
    return np.sqrt(MSE(Y, Y_true))

def MAPE(Y, Y_true):
    return np.mean(np.fabs((Y_true - Y) / Y_true)) * 100

def generate_weights(X):
    return np.matrix(np.random.randn(X.shape[1])/np.sqrt(np.sum(X.shape[1])))

def grad_loss(X, Y, B):
    return ((X * B.T - Y).T * X) / X.shape[0]

def update_weights(B, loss, lr):
    return B - lr * loss

def grad_ridge_reg_loss(B, l):
    return np.mean(B) * l

def grad_lasso_reg_loss(B, l):
    return np.sign(B) * l