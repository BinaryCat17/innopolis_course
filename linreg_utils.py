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

# binary cross entropy
def BCE(Y, Y_true):
    Y= Y.T
    eps = 1e-6
    return -(Y_true*np.log(Y + eps)+(1 - Y_true) * np.log(1 - Y + eps)).mean()

def generate_weights(X):
    return np.matrix(np.random.randn(X.shape[1])/np.sqrt(np.sum(X.shape[1])))

def grad_loss(Y, X_t, Y_t):
    return ((Y - Y_t).T * X_t) / X_t.shape[0]

def update_weights(B, loss, lr):
    return B - lr * loss

def ridge_reg_loss(B, l):
    return (l/2)*np.sum(B[1:])

def der_ridge_reg_loss(B, l):
    term = 2 * B[:, 1:] * l
    return np.concatenate([np.zeros((1, 1)), term], axis=1)

def lasso_reg_loss(B, l):
    return l*np.sum(np.abs(B[1:]))

def der_lasso_reg_loss(B, l):
    term = np.sign(B[:, 1:]) * l
    return np.concatenate([np.zeros((1, 1)), term], axis=1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))     

def to_class(logit, threshold = 0.7):
    return (logit>=threshold)*1
